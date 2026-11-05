"""
Loss functions for Multi-Class CT Hemorrhage Segmentation.

Combines Weighted Cross-Entropy / Focal Cross-Entropy Loss with Multiclass Soft Dice Loss.
Class frequencies and weights are strictly derived from the training set only.
"""

from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights_from_manifest(
    train_manifest_csv: str,
    dataset_dir: str = "Dataset",
    num_classes: int = 6,
    weight_power: float = 0.5,
    max_weight: float = 75.0,
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from the TRAINING set masks only.
    Smooths extreme rarity using square-root power.
    
    Capped at max_weight (default 75.0) to allow the rarest class (EDH, ~0.018% of voxels)
    to receive its true relative gradient weight (~70.8) without being prematurely capped
    at the same weight as much more common bleeds like IPH (which has ~6.5x more pixels).
    
    Formula:
        freq_c = total_pixels_c / total_all_pixels
        raw_weight_c = (1.0 / freq_c) ** weight_power
        weight_c = min(raw_weight_c / raw_weight_bg, max_weight)
    """
    import os
    import json
    import pandas as pd
    import nibabel as nib
    from src.data.split import BHSD_TO_TARGET

    manifest_dir = os.path.dirname(train_manifest_csv) or "data_splits"
    counts_cache = os.path.join(manifest_dir, "train_class_counts.json")

    if os.path.exists(counts_cache):
        with open(counts_cache, "r") as f:
            cached_counts = json.load(f)
        pixel_counts = np.array([cached_counts[str(c)] for c in range(num_classes)], dtype=np.int64)
    else:
        df = pd.read_csv(train_manifest_csv)
        gt_dir = os.path.join(dataset_dir, "labeled", "ground_truths")
        pixel_counts = np.zeros(num_classes, dtype=np.int64)

        for fname in df["filename"]:
            gt_path = os.path.join(gt_dir, fname)
            gt_nii = nib.load(gt_path)
            gt_data = np.asanyarray(gt_nii.dataobj)
            for raw_c, target_c in BHSD_TO_TARGET.items():
                pixel_counts[target_c] += np.sum(gt_data == raw_c)
            gt_nii.uncache()
            del gt_nii

        try:
            with open(counts_cache, "w") as f:
                json.dump({str(c): int(pixel_counts[c]) for c in range(num_classes)}, f)
        except Exception:
            pass

    total_pixels = pixel_counts.sum()
    freqs = pixel_counts / float(total_pixels)
    
    # Avoid div by zero
    freqs = np.maximum(freqs, 1e-8)
    
    # Inverse square-root frequencies
    raw_weights = (1.0 / freqs) ** weight_power
    # Normalize with respect to background weight
    weights = raw_weights / raw_weights[0]
    # Cap maximum weight to prevent gradient explosion on tiny lesions
    weights = np.clip(weights, 1.0, max_weight)
    
    print("\n=== TRAINING-DERIVED CLASS WEIGHTS ===")
    from src.data.split import TARGET_NAMES
    for c in range(num_classes):
        print(f"  Class {c} ({TARGET_NAMES[c]}): Pixel count={pixel_counts[c]:,} ({freqs[c]*100:.4f}%), Weight={weights[c]:.2f}")
        
    return torch.tensor(weights, dtype=torch.float32)


class WeightedFocalCrossEntropyLoss(nn.Module):
    """
    Focal Cross-Entropy Loss with class weighting.
    Formula: FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Down-weights easy background pixels (p_t ~ 1.0) so the model focuses its gradient
    budget on difficult, subtle lesion boundaries (especially rare EDH and thin SAH).
    If gamma=0, acts as standard Weighted Cross-Entropy.
    """
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy per pixel without reduction
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        if self.gamma > 0:
            # pt is probability of true class
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
            focal_modulation = (1.0 - pt) ** self.gamma
            loss = focal_modulation * ce_loss
        else:
            loss = ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MulticlassDiceLoss(nn.Module):
    """
    Per-Slice Multiclass Soft Dice & Focal Tversky Loss.
    
    Key Improvements:
    1. Computes Dice/Tversky per slice along spatial dimensions (H, W).
       Prevents empty slices in a multi-slice batch from diluting the denominator
       and drowning out gradients on slices containing small lesions (especially EDH).
    2. Tversky formulation (alpha=0.3, beta=0.7) heavily penalizes False Negatives (missed bleeds)
       2.33x more than False Positives, directly preventing the model from collapsing to 0.0000.
    3. Excludes dominant background class by default (c=1..5).
    """
    def __init__(
        self,
        num_classes: int = 6,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
        include_background: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, H, W] with integer values 0..C-1
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        start_idx = 0 if self.include_background else 1
        class_losses = []
        weights = []

        for c in range(start_idx, self.num_classes):
            p_c = probs[:, c]  # [B, H, W]
            t_c = targets_one_hot[:, c]  # [B, H, W]

            tp = (p_c * t_c).sum(dim=(1, 2))  # [B]
            fp = (p_c * (1.0 - t_c)).sum(dim=(1, 2))  # [B]
            fn = ((1.0 - p_c) * t_c).sum(dim=(1, 2))  # [B]

            # Per-slice Tversky Index
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)  # [B]
            focal_tv = torch.pow((1.0 - tversky).clamp(min=1e-6), self.gamma)  # [B]

            # Prioritize slices containing the lesion (2x weight)
            has_lesion = (t_c.sum(dim=(1, 2)) > 0).float()
            slice_weight = 1.0 + 2.0 * has_lesion
            c_loss = (focal_tv * slice_weight).sum() / slice_weight.sum().clamp(min=1.0)
            class_losses.append(c_loss)

            if self.class_weights is not None and c < len(self.class_weights):
                weights.append(self.class_weights[c])
            else:
                weights.append(1.0)

        losses_t = torch.stack(class_losses)
        if self.class_weights is not None:
            w_t = torch.tensor(weights, dtype=losses_t.dtype, device=losses_t.device)
            w_t = w_t / w_t.sum()
            return (losses_t * w_t).sum()
        return losses_t.mean()


class CombinedDiceAndCrossEntropyLoss(nn.Module):
    """
    Combined objective: L = alpha * Weighted_Focal_CE + beta * Per_Slice_Focal_Tversky_Dice
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha_ce: float = 1.0,
        beta_dice: float = 2.0,
        focal_gamma: float = 1.5,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        num_classes: int = 6,
        smooth: float = 1.0,
        include_background_dice: bool = False,
    ):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_dice = beta_dice
        self.ce_loss = WeightedFocalCrossEntropyLoss(weight=class_weights, gamma=focal_gamma)
        
        # Priority weighting for foreground classes in Dice loss:
        # Boost rare classes (EDH: 1.8x, SDH: 1.4x, SAH: 1.3x)
        fg_dice_weights = torch.tensor([1.0, 1.0, 1.3, 1.8, 1.4, 1.0]) if num_classes == 6 else None
        self.dice_loss = MulticlassDiceLoss(
            num_classes=num_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=1.33,
            smooth=smooth,
            include_background=include_background_dice,
            class_weights=fg_dice_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.alpha_ce * ce + self.beta_dice * dice
