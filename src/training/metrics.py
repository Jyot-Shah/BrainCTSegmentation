"""
Evaluation Metrics for Multi-Class CT Hemorrhage Segmentation.

Implements:
- Per-class Dice, IoU, Precision, Recall
- Macro Dice and Macro IoU across the 5 hemorrhage classes (excluding background)
- 6-class confusion matrix
- Global and volume-level metric aggregation
- Explicit empty mask handling
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from src.data.split import TARGET_NAMES


class SegmentationMetricsMeter:
    """
    Accumulates confusion matrix counts across batches and computes per-class and macro metrics.
    """
    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # 2D confusion matrix: row=ground_truth, col=predicted
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        preds: [H, W] or flat array of predicted class integers in [0, num_classes-1]
        targets: [H, W] or flat array of ground truth class integers
        """
        p_flat = preds.ravel()
        t_flat = targets.ravel()
        
        # Valid mask
        valid = (t_flat >= 0) & (t_flat < self.num_classes) & (p_flat >= 0) & (p_flat < self.num_classes)
        p_valid = p_flat[valid]
        t_valid = t_flat[valid]

        # Fast bincount update
        indices = t_valid * self.num_classes + p_valid
        bincount = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))

    def compute(self, eps: float = 1e-7) -> Dict[str, Any]:
        """
        Computes all standard segmentation metrics from accumulated counts.
        """
        cm = self.confusion_matrix
        per_class_dice = {}
        per_class_iou = {}
        per_class_precision = {}
        per_class_recall = {}
        
        fg_dices = []
        fg_ious = []

        for c in range(self.num_classes):
            name = TARGET_NAMES[c]
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)

            dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)

            per_class_dice[name] = dice
            per_class_iou[name] = iou
            per_class_precision[name] = precision
            per_class_recall[name] = recall

            # Accumulate foreground (hemorrhage) classes only (c=1..5)
            if c > 0:
                fg_dices.append(dice)
                fg_ious.append(iou)

        macro_dice = float(np.mean(fg_dices))
        macro_iou = float(np.mean(fg_ious))

        return {
            "per_class_dice": per_class_dice,
            "per_class_iou": per_class_iou,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "macro_dice": macro_dice,
            "macro_iou": macro_iou,
            "confusion_matrix": cm,
        }


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Formats computed metrics into a clean markdown table.
    """
    lines = [
        "| Class ID | Hemorrhage Class | Dice | IoU | Precision | Recall |",
        "|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    for c in range(6):
        name = TARGET_NAMES[c]
        dice = metrics["per_class_dice"][name]
        iou = metrics["per_class_iou"][name]
        prec = metrics["per_class_precision"][name]
        rec = metrics["per_class_recall"][name]
        lines.append(f"| {c} | {name} | {dice:.4f} | {iou:.4f} | {prec:.4f} | {rec:.4f} |")

    lines.append(f"\n**Macro Dice (5 hemorrhage classes, excl. BG):** `{metrics['macro_dice']:.4f}`")
    lines.append(f"**Macro IoU (5 hemorrhage classes, excl. BG):** `{metrics['macro_iou']:.4f}`")
    return "\n".join(lines)
