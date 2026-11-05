"""
Comprehensive Evaluation Pipeline for 2.5D Brain CT Hemorrhage Segmentation.

Produces:
1. Volume-level and slice-level metrics (Dice, IoU, Precision, Recall).
2. Macro Dice & Macro IoU across the 5 hemorrhage classes (excluding background).
3. 6-Class Confusion Matrix plot (normalized and raw).
4. Training curve plots (Loss, Macro Dice, Per-Class Dice).
5. Ground truth vs Prediction overlay comparison figures (successes and failure modes).
6. Metrics saved to CSV and markdown report.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import SegmentationMetricsMeter, format_metrics_table
from src.data.split import TARGET_NAMES
from src.data.visualize import CLASS_COLORS, OVERLAY_CMAP


def plot_training_curves(history_csv: str, save_path: str = "outputs/figures/training_curves.png"):
    """
    Plots training loss, validation loss, validation macro Dice, and per-class Dice curves.
    """
    if not os.path.exists(history_csv):
        print(f"[WARN] History CSV not found at {history_csv}, skipping curve plotting.")
        return

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss Curves
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="crimson", lw=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Macro Dice Curve
    axes[1].plot(df["epoch"], df["val_macro_dice"], label="Val Macro Dice (5 Hemorrhages)", color="darkgreen", lw=2.5)
    if "val_macro_iou" in df:
        axes[1].plot(df["epoch"], df["val_macro_iou"], label="Val Macro IoU", color="darkorange", lw=1.8, linestyle="--")
    axes[1].set_title("Validation Macro Dice & IoU", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Per-Class Dice Curves
    palette = ["#1f77b4", "#e377c2", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, name in enumerate(TARGET_NAMES[1:]):  # Hemorrhage classes 1..5
        col = f"val_dice_{name}"
        if col in df:
            axes[2].plot(df["epoch"], df[col], label=name, lw=1.8, color=palette[i % len(palette)])
    axes[2].set_title("Per-Class Validation Dice", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice Score")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Training curves saved to: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = TARGET_NAMES,
    normalize: bool = True,
    save_path: str = "outputs/figures/confusion_matrix.png",
):
    """
    Renders and saves the 6-class pixel confusion matrix.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(np.float64) / row_sums
        mat_to_show = cm_norm
        fmt = ".2f"
        title = "Normalized Confusion Matrix (Recall per Class)"
    else:
        mat_to_show = cm
        fmt = "d"
        title = "Raw Pixel Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat_to_show, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Annotate numbers in each cell
    thresh = mat_to_show.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = mat_to_show[i, j]
            text = f"{val:{fmt}}" if not normalize else f"{val*100:.1f}%"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix saved to: {save_path}")


def evaluate_model_on_dataset(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = "outputs/metrics",
    figure_dir: str = "outputs/figures",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full evaluation pass over the validation dataset:
    - Calculates per-class Dice, IoU, Precision, Recall
    - Macro Dice and Macro IoU across 5 hemorrhage classes
    - Confusion Matrix
    - Overlays of predictions vs ground truth
    """
    model.eval()
    model.to(device)
    meter = SegmentationMetricsMeter(num_classes=6)

    # Save representative comparison visual examples
    saved_overlays = 0
    max_overlays = 6

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("\n[INFO] Running evaluation on validation partition...")
    with torch.no_grad():
        for step, (images, masks, sample_info) in enumerate(tqdm(val_loader, desc="Evaluating")):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks_np = masks.numpy()
            meter.update(preds, masks_np)

            # Generate overlay figures for slices containing hemorrhages
            for b in range(images.shape[0]):
                gt_slice = masks_np[b]
                pred_slice = preds[b]
                
                # Check if slice has hemorrhage in GT or Prediction
                if (gt_slice > 0).any() and saved_overlays < max_overlays:
                    saved_overlays += 1
                    # Extract middle slice brain window (channel 3)
                    # Channels: 0-2 (prev slice), 3-5 (curr slice: Brain, Subdural, Bone)
                    brain_win = images[b, 3].cpu().numpy()
                    # Denormalize brain window for display
                    # ImageNet norm: (x - mean) / std -> x = norm * std + mean
                    brain_win = brain_win * 0.485 + 0.485
                    brain_win = np.clip(brain_win, 0.0, 1.0)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(brain_win, cmap="gray")
                    axes[0].set_title("CT Brain Window", fontsize=11)
                    axes[0].axis("off")

                    # GT overlay
                    axes[1].imshow(brain_win, cmap="gray")
                    axes[1].imshow(gt_slice, cmap=OVERLAY_CMAP, vmin=0, vmax=5, interpolation="nearest")
                    axes[1].set_title("Ground Truth", fontsize=11)
                    axes[1].axis("off")

                    # Prediction overlay
                    axes[2].imshow(brain_win, cmap="gray")
                    axes[2].imshow(pred_slice, cmap=OVERLAY_CMAP, vmin=0, vmax=5, interpolation="nearest")
                    axes[2].set_title("DeepLabV3+ Prediction", fontsize=11)
                    axes[2].axis("off")

                    # Add legend
                    classes_present = np.union1d(np.unique(gt_slice), np.unique(pred_slice))
                    patches = []
                    for c_idx in range(1, 6):
                        if c_idx in classes_present:
                            p = mpatches.Patch(color=CLASS_COLORS[c_idx][:3], label=TARGET_NAMES[c_idx])
                            patches.append(p)
                    if patches:
                        axes[2].legend(handles=patches, loc="lower right", fontsize=8)

                    overlay_file = os.path.join(figure_dir, f"prediction_overlay_{saved_overlays}.png")
                    plt.tight_layout()
                    plt.savefig(overlay_file, dpi=150, bbox_inches="tight")
                    plt.close()

    metrics = meter.compute()
    table_md = format_metrics_table(metrics)
    print("\n" + table_md)

    # Save metrics to CSV
    metrics_records = []
    for c in range(6):
        name = TARGET_NAMES[c]
        metrics_records.append({
            "class_id": c,
            "class_name": name,
            "dice": metrics["per_class_dice"][name],
            "iou": metrics["per_class_iou"][name],
            "precision": metrics["per_class_precision"][name],
            "recall": metrics["per_class_recall"][name],
        })
    metrics_df = pd.DataFrame(metrics_records)
    csv_path = os.path.join(output_dir, "validation_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    summary_df = pd.DataFrame([{
        "macro_dice_hemorrhages": metrics["macro_dice"],
        "macro_iou_hemorrhages": metrics["macro_iou"],
    }])
    summary_df.to_csv(os.path.join(output_dir, "macro_summary.csv"), index=False)

    # Plot Confusion Matrix
    cm_path = os.path.join(figure_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)

    # Save report
    with open(os.path.join(output_dir, "evaluation_report.md"), "w") as f:
        f.write("# Brain CT Hemorrhage Segmentation Baseline Evaluation\n\n")
        f.write(table_md)
        f.write("\n\n## Confusion Matrix\n![Confusion Matrix](../figures/confusion_matrix.png)\n")

    return metrics
