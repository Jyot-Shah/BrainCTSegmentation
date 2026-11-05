"""
Active Learning Visualization and Publication Graphics Generator.

Implements:
- The Headline Two-Curve Label-Efficiency Plot (Active Learning vs. Random Selection).
- Per-class label-efficiency curves (EDH, IPH, SAH, SDH, IVH).
- Epistemic uncertainty spatial overlays on CT slices (demonstrating boundary concentration).
- Entropy distribution shift / uncertainty compression plots (Round 0 vs Round 2).
- Stratified Sampling vs Pure Top-K entropy distribution comparison (Week 10 deliverable).
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TARGET_NAMES = ["Background", "IPH", "SAH", "EDH", "SDH", "IVH"]
TARGET_COLORS = ["#000000", "#e11d48", "#f59e0b", "#eab308", "#8b5cf6", "#06b6d4"]


def plot_label_efficiency_curves(
    history_df: pd.DataFrame,
    save_path: str = "outputs/active_learning/figures/label_efficiency_curves.png",
) -> None:
    """
    Plots the headline two-curve label efficiency benchmark comparison:
    Top: Macro Validation Dice vs. Number of newly annotated slices.
    Bottom: 5-panel per-class label efficiency breakdown.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    al_data = history_df[history_df["strategy"].str.contains("Active")].sort_values("total_annotated_slices")
    rnd_data = history_df[history_df["strategy"].str.contains("Random")].sort_values("total_annotated_slices")

    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.3)

    # 1. Main Macro Dice Label-Efficiency Curve (Spans top row)
    ax_main = fig.add_subplot(gs[0, :])
    x_al = al_data["total_annotated_slices"].values
    y_al = al_data["macro_dice"].values
    x_rnd = rnd_data["total_annotated_slices"].values
    y_rnd = rnd_data["macro_dice"].values

    ax_main.plot(x_al, y_al, marker="o", markersize=8, color="#2563eb", linewidth=2.5, label="Active Learning (Stratified Uncertainty)")
    ax_main.plot(x_rnd, y_rnd, marker="s", markersize=7, color="#64748b", linewidth=2.0, linestyle="--", label="Random Selection (Passive Control)")

    # Fill area between curves
    if len(x_al) == len(x_rnd) and (x_al == x_rnd).all():
        ax_main.fill_between(x_al, y_al, y_rnd, color="#3b82f6", alpha=0.15, label="Active Learning Efficiency Gap")

    # Annotate final delta
    if len(y_al) > 1 and len(y_rnd) > 1:
        final_delta = y_al[-1] - y_rnd[-1]
        ax_main.annotate(
            f"AL Advantage: {final_delta:+.4f} Macro Dice\n({final_delta*100:+.2f}%)",
            xy=(x_al[-1], y_al[-1]),
            xytext=(x_al[-1] - 30, y_al[-1] + 0.015),
            arrowprops=dict(facecolor="#1e40af", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9.5,
            fontweight="bold",
            color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="#2563eb", facecolor="#eff6ff"),
        )

    ax_main.set_title("Label-Efficiency Curve: Active Learning vs. Random Baseline (Macro Validation Dice)", fontsize=13, fontweight="bold", pad=10)
    ax_main.set_xlabel("Cumulative Newly Annotated Slices Added to Training Set", fontsize=10, fontweight="bold")
    ax_main.set_ylabel("Held-Out Macro Validation Dice", fontsize=10, fontweight="bold")
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="lower right", frameon=True, fontsize=10)

    # 2. Per-Class Subplots
    classes = [("IPH", "iph_dice", "#e11d48"), ("SAH", "sah_dice", "#f59e0b"), ("EDH", "edh_dice", "#eab308"), ("SDH", "sdh_dice", "#8b5cf6"), ("IVH", "ivh_dice", "#06b6d4")]

    for idx, (c_name, c_col, c_color) in enumerate(classes):
        ax = fig.add_subplot(gs[1, idx])
        ax.plot(x_al, al_data[c_col].values, marker="o", markersize=5, color=c_color, linewidth=2.0, label="AL")
        ax.plot(x_rnd, rnd_data[c_col].values, marker="s", markersize=4, color="#94a3b8", linewidth=1.5, linestyle="--", label="Random")

        ax.set_title(f"{c_name} Subtype", fontsize=10.5, fontweight="bold", color=c_color)
        ax.set_xlabel("Annotated Slices", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("Validation Dice", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=7.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[FIGURE] Saved Label-Efficiency Two-Curve figure to: {save_path}")


def plot_stratified_vs_topk(
    stratified_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    save_path: str = "outputs/active_learning/figures/stratified_vs_topk.png",
) -> None:
    """
    Plots entropy distribution of pure Top-K vs Stratified selection (Week 10 deliverable).
    Demonstrates that pure Top-K clusters at the entropy ceiling while Stratified spans
    high, medium, and boundary uncertainty bands.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=300)

    plt.hist(
        topk_df["mean_fg_entropy"],
        bins=25,
        alpha=0.55,
        color="#dc2626",
        label="Pure Top-K Selection (Clusters at ceiling)",
        density=True,
    )
    plt.hist(
        stratified_df["mean_fg_entropy"],
        bins=25,
        alpha=0.65,
        color="#2563eb",
        label="Stratified Selection (Bands A, B, C)",
        density=True,
    )

    plt.axvline(0.50, color="#1e40af", linestyle=":", linewidth=1.5, label="Band A / B Boundary (0.50)")
    plt.axvline(0.30, color="#60a5fa", linestyle=":", linewidth=1.5, label="Band B / C Boundary (0.30)")

    plt.title("Stratified Selection vs. Pure Top-K Entropy Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Predictive Entropy H(p)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, fontsize=8.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[FIGURE] Saved Stratified vs Top-K distribution figure to: {save_path}")


def plot_entropy_drift(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    save_path: str = "outputs/active_learning/figures/entropy_drift.png",
) -> None:
    """
    Plots epistemic uncertainty compression / drift across active learning rounds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)

    plt.hist(before_df["mean_fg_entropy"], bins=30, alpha=0.6, color="#64748b", label="Round 0 (Initial Pool)", density=True)
    plt.hist(after_df["mean_fg_entropy"], bins=30, alpha=0.6, color="#2563eb", label="Round 2 (After Active Acquisition)", density=True)

    plt.axvline(before_df["mean_fg_entropy"].median(), color="#475569", linestyle="--", linewidth=1.5, label=f"Median R0: {before_df['mean_fg_entropy'].median():.3f}")
    plt.axvline(after_df["mean_fg_entropy"].median(), color="#1d4ed8", linestyle="--", linewidth=1.5, label=f"Median R2: {after_df['mean_fg_entropy'].median():.3f}")

    plt.title("Epistemic Uncertainty Compression: Pool Entropy Shift", fontsize=11, fontweight="bold")
    plt.xlabel("Foreground Predictive Entropy", fontsize=9.5)
    plt.ylabel("Density", fontsize=9.5)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, fontsize=8.5)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[FIGURE] Saved Entropy Compression figure to: {save_path}")


def plot_uncertainty_overlay(
    ct_image: np.ndarray,
    ground_truth: np.ndarray,
    pred_mask: np.ndarray,
    entropy_map: np.ndarray,
    slice_meta: Dict[str, Any],
    save_path: str = "outputs/active_learning/figures/uncertainty_overlay.png",
) -> None:
    """
    Generates a 4-panel visual validation figure demonstrating:
    Panel 1: Raw CT (Brain Window)
    Panel 2: Ground Truth Annotations
    Panel 3: DeepLabV3+ Predicted Segmentation
    Panel 4: Epistemic Uncertainty Heatmap (Concentrating at lesion boundaries)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

    # 1. Raw CT Brain Window
    axes[0].imshow(ct_image, cmap="bone")
    axes[0].set_title("Raw CT (Brain Window)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth
    axes[1].imshow(ct_image, cmap="bone")
    if ground_truth is not None and ground_truth.max() > 0:
        masked_gt = np.ma.masked_where(ground_truth == 0, ground_truth)
        axes[1].imshow(masked_gt, cmap="tab10", vmin=1, vmax=5, alpha=0.7)
    axes[1].set_title("Ground Truth Mask", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3. Model Prediction
    axes[2].imshow(ct_image, cmap="bone")
    if pred_mask is not None and pred_mask.max() > 0:
        masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[2].imshow(masked_pred, cmap="tab10", vmin=1, vmax=5, alpha=0.7)
    axes[2].set_title("DeepLabV3+ Prediction", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # 4. Uncertainty Heatmap
    axes[3].imshow(ct_image, cmap="bone", alpha=0.6)
    im = axes[3].imshow(entropy_map, cmap="inferno", alpha=0.75)
    axes[3].set_title("Epistemic Uncertainty (Entropy)", fontsize=10, fontweight="bold")
    axes[3].axis("off")
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Predictive Entropy H(p)", fontsize=8)

    fname = slice_meta.get("filename", "N/A")
    s_idx = slice_meta.get("slice_idx", 0)
    cat = slice_meta.get("category", "")
    title = f"Active Learning Case: {fname} (Slice {s_idx})"
    if cat:
        title += f" [{cat}]"
    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[FIGURE] Saved Uncertainty Overlay inspection to: {save_path}")
