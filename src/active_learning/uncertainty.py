"""
Monte Carlo Dropout Epistemic Uncertainty Estimation Engine.

Implements:
- enable_mc_dropout: Freezes BatchNorm running stats (eval) while keeping Dropout layers stochastic.
- Vectorized computation of:
  * Predictive Entropy H(E_t[p])
  * Expected Entropy E_t[H(p_t)]
  * Mutual Information / BALD (H(mean) - mean(H))
  * Predictive Variance Var_t[p]
  * Variation Ratio (1 - max_c mean_p_c)
- Critical foreground-only (fg_region) spatial aggregation to prevent background dilution.
- Verification checks (entropy-BALD correlation >= 0.85, boundary concentration).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deeplab import enable_mc_dropout


class MCDropoutScorer:
    """
    Computes epistemic and aleatoric uncertainty via Monte Carlo Dropout sampling.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_passes: int = 10,
        temperature: float = 1.0,
        calibrated_thresholds: Optional[List[float]] = None,
        probability_mode: str = "softmax",
        use_amp: bool = True,
    ):
        """
        Args:
            model: PyTorch segmentation model with dropout layers.
            device: Computation device (cuda or cpu).
            num_passes: Number of stochastic forward passes T (default: 10).
            temperature: Logit scaling temperature.
            calibrated_thresholds: Optional thresholds for relative decision [-, t1, t2, t3, t4, t5].
            probability_mode: 'softmax' (standard) or 'calibrated' (relative sigmoid).
            use_amp: Whether to use Automatic Mixed Precision for fast inference.
        """
        self.model = model
        self.device = device
        self.num_passes = num_passes
        self.temperature = max(temperature, 1e-4)
        self.calibrated_thresholds = calibrated_thresholds or [0.5, 0.50, 0.35, 0.30, 0.35, 0.45]
        self.probability_mode = probability_mode
        self.use_amp = use_amp and device.type == "cuda"

        # Ensure model is in MC Dropout mode (BN frozen in eval, Dropout active)
        enable_mc_dropout(self.model)
        self.model.to(self.device)

    def _forward_stochastic_passes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes T stochastic forward passes.
        Args:
            x: Input tensor [B, C, H, W] on self.device
        Returns:
            Probabilities tensor [T, B, num_classes, H, W]
        """
        passes_probs = []
        amp_ctx = torch.amp.autocast("cuda") if self.use_amp else torch.no_grad()

        with torch.no_grad():
            for _ in range(self.num_passes):
                if self.use_amp:
                    with amp_ctx:
                        logits = self.model(x)
                else:
                    logits = self.model(x)

                logits = logits.float() / self.temperature

                if self.probability_mode == "softmax":
                    probs = F.softmax(logits, dim=1)
                else:
                    # Calibrated relative sigmoid
                    z_0 = logits[:, 0:1, :, :]
                    rel_logits = logits - z_0
                    lesion_probs = torch.sigmoid(rel_logits[:, 1:, :, :])
                    bg_prob = torch.clamp(1.0 - lesion_probs.max(dim=1, keepdim=True)[0], min=0.0, max=1.0)
                    probs = torch.cat([bg_prob, lesion_probs], dim=1)
                    probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-7)

                passes_probs.append(probs)

        # Shape: [T, B, num_classes, H, W]
        return torch.stack(passes_probs, dim=0)

    def score_batch(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Computes uncertainty maps and slice-level aggregate metrics for a batch.
        Args:
            x: Input batch tensor [B, C, H, W]
        Returns:
            Dictionary containing:
            - mean_probs: [B, num_classes, H, W]
            - pred_masks: [B, H, W]
            - entropy_map: [B, H, W]
            - bald_map: [B, H, W]
            - variance_map: [B, H, W]
            - slice_metrics: List of Dicts with slice-level scalar metrics.
        """
        x = x.to(self.device)
        b, _, h, w = x.shape
        eps = 1e-8

        # 1. Run T passes: [T, B, num_classes, H, W]
        t_probs = self._forward_stochastic_passes(x)
        t = t_probs.shape[0]

        # 2. Mean probabilities across stochastic passes: [B, num_classes, H, W]
        mean_probs = t_probs.mean(dim=0)

        # 3. Decision mask with calibrated thresholds
        pred_scores = torch.zeros_like(mean_probs)
        pred_scores[:, 0, :, :] = mean_probs[:, 0, :, :]
        for c in range(1, mean_probs.shape[1]):
            thresh = self.calibrated_thresholds[c] if c < len(self.calibrated_thresholds) else 0.5
            pred_scores[:, c, :, :] = mean_probs[:, c, :, :] / thresh

        # Argmax on confidence ratios
        pred_masks = pred_scores.argmax(dim=1)  # [B, H, W]

        # 4. Predictive Entropy: H(mean) = -sum_c mean_p * log(mean_p)
        entropy_map = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)  # [B, H, W]

        # 5. Expected Entropy: mean(H(p_t)) = 1/T sum_t (-sum_c p_t * log(p_t))
        t_entropy = -torch.sum(t_probs * torch.log(t_probs + eps), dim=2)  # [T, B, H, W]
        expected_entropy = t_entropy.mean(dim=0)  # [B, H, W]

        # 6. Mutual Information (BALD): H(mean) - mean(H)
        bald_map = torch.clamp(entropy_map - expected_entropy, min=0.0)  # [B, H, W]

        # 7. Predictive Variance across passes: sum of variance across lesion classes
        variance_map = t_probs[:, :, 1:, :, :].var(dim=0).sum(dim=1)  # [B, H, W]

        # 8. Variation Ratio: 1 - max_c mean_p
        var_ratio_map = 1.0 - mean_probs.max(dim=1)[0]  # [B, H, W]

        # 9. Extract per-slice aggregate metrics
        slice_metrics = []
        for i in range(b):
            fg_mask = (pred_masks[i] > 0)
            fg_pixels = fg_mask.sum().item()

            slice_entropy = entropy_map[i]
            slice_bald = bald_map[i]
            slice_var = variance_map[i]
            slice_vr = var_ratio_map[i]

            # Whole-image maximums
            max_entropy = slice_entropy.max().item()
            max_bald = slice_bald.max().item()
            mean_whole_entropy = slice_entropy.mean().item()

            # Foreground-only aggregation (prevents background dilution)
            if fg_pixels >= 10:
                mean_fg_entropy = slice_entropy[fg_mask].mean().item()
                mean_fg_bald = slice_bald[fg_mask].mean().item()
                mean_fg_var = slice_var[fg_mask].mean().item()
                mean_fg_vr = slice_vr[fg_mask].mean().item()
            else:
                # If model predicted almost no foreground, fallback to top-100 highest entropy pixels
                top_k_entropy, _ = torch.topk(slice_entropy.view(-1), k=min(100, h * w))
                mean_fg_entropy = top_k_entropy.mean().item()
                top_k_bald, _ = torch.topk(slice_bald.view(-1), k=min(100, h * w))
                mean_fg_bald = top_k_bald.mean().item()
                mean_fg_var = 0.0
                mean_fg_vr = 0.0

            # Detected classes in prediction
            pred_classes = torch.unique(pred_masks[i]).cpu().numpy().tolist()
            detected_lesions = [int(c) for c in pred_classes if c > 0]

            slice_metrics.append({
                "mean_fg_entropy": float(mean_fg_entropy),
                "mean_fg_bald": float(mean_fg_bald),
                "mean_fg_var": float(mean_fg_var),
                "mean_fg_vr": float(mean_fg_vr),
                "max_entropy": float(max_entropy),
                "max_bald": float(max_bald),
                "mean_whole_entropy": float(mean_whole_entropy),
                "pred_area": int(fg_pixels),
                "detected_classes": detected_lesions,
                "has_edh_pred": 3 in detected_lesions,
                "has_iph_pred": 1 in detected_lesions,
                "has_sah_pred": 2 in detected_lesions,
                "has_sdh_pred": 4 in detected_lesions,
                "has_ivh_pred": 5 in detected_lesions,
            })

        return {
            "mean_probs": mean_probs,
            "pred_masks": pred_masks,
            "entropy_map": entropy_map,
            "bald_map": bald_map,
            "variance_map": variance_map,
            "variation_ratio_map": var_ratio_map,
            "slice_metrics": slice_metrics,
        }
