"""
Visualization utility for 2.5D Multi-Window Brain CT and Ground Truth Overlays.

Renders Brain, Subdural, and Bone windows with multi-class hemorrhage overlays
and a clear clinical color legend.
"""

import os
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch

from src.data.transforms import apply_3_windows
from src.data.split import TARGET_NAMES

# High-contrast clinical color palette for the 6 target classes:
# 0: Background (transparent)
# 1: IPH (Red)
# 2: SAH (Cyan)
# 3: EDH (Yellow)
# 4: SDH (Orange)
# 5: IVH (Magenta)
CLASS_COLORS = [
    (0.0, 0.0, 0.0, 0.0),       # 0: BG (transparent)
    (0.9, 0.1, 0.1, 0.65),      # 1: IPH (hyperdense red)
    (0.1, 0.8, 0.8, 0.70),      # 2: SAH (cyan)
    (1.0, 0.85, 0.0, 0.75),     # 3: EDH (bright yellow)
    (1.0, 0.5, 0.0, 0.70),      # 4: SDH (orange)
    (0.8, 0.1, 0.8, 0.70),      # 5: IVH (magenta)
]

OVERLAY_CMAP = ListedColormap(CLASS_COLORS)


def render_slice_with_windows_and_mask(
    slice_hu: np.ndarray,
    mask: np.ndarray,
    title: str = "CT Slice Inspection",
    save_path: Optional[str] = None,
):
    """
    Plots Brain, Subdural, and Bone windows side-by-side along with the GT overlay.
    """
    windows = apply_3_windows(slice_hu)  # [3, H, W] in [0, 1]
    w_brain = windows[0]
    w_subdural = windows[1]
    w_bone = windows[2]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # 1. Brain window
    axes[0].imshow(w_brain, cmap="gray")
    axes[0].set_title("Brain Window\n(WL=40, WW=80)", fontsize=11)
    axes[0].axis("off")

    # 2. Subdural window
    axes[1].imshow(w_subdural, cmap="gray")
    axes[1].set_title("Subdural Window\n(WL=75, WW=215)", fontsize=11)
    axes[1].axis("off")

    # 3. Bone window
    axes[2].imshow(w_bone, cmap="gray")
    axes[2].set_title("Bone Window\n(WL=400, WW=1800)", fontsize=11)
    axes[2].axis("off")

    # 4. Brain window with Ground Truth Overlay
    axes[3].imshow(w_brain, cmap="gray")
    overlay = axes[3].imshow(mask, cmap=OVERLAY_CMAP, vmin=0, vmax=5, interpolation="nearest")
    axes[3].set_title("Ground Truth Overlay\n(Anatomy Alignment)", fontsize=11)
    axes[3].axis("off")

    # Build Legend
    present_classes = np.unique(mask)
    legend_patches = []
    for c_idx in range(1, 6):
        if c_idx in present_classes:
            name = f"{TARGET_NAMES[c_idx]} (Class {c_idx})"
            patch = mpatches.Patch(color=CLASS_COLORS[c_idx][:3], label=name)
            legend_patches.append(patch)
            
    if legend_patches:
        axes[3].legend(handles=legend_patches, loc="lower right", framealpha=0.85, fontsize=8)
    else:
        axes[3].text(
            15, 30, "No hemorrhage (Background only)", 
            color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.6)
        )

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_dataset_samples(
    dataset,
    num_samples: int = 5,
    save_dir: str = "outputs/figures",
):
    """
    Finds and visualizes representative slices containing hemorrhages from the dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    
    # Prioritize slices that contain hemorrhages
    for idx in range(len(dataset)):
        meta = dataset.slice_indices[idx]
        if meta["has_foreground"]:
            # Retrieve raw HU slice
            fname = meta["filename"]
            z = meta["slice_idx"]
            img_vol, gt_vol = dataset._load_volume(fname)
            
            slice_hu = img_vol[:, :, z]
            mask = gt_vol[:, :, z]
            
            classes_str = ", ".join([TARGET_NAMES[c] for c in meta["target_classes"] if c > 0])
            title = f"{fname} - Slice {z}/{meta['num_slices']} [{classes_str}]"
            save_path = os.path.join(save_dir, f"sample_{count+1}_{fname}_z{z}.png")
            
            render_slice_with_windows_and_mask(slice_hu, mask, title=title, save_path=save_path)
            print(f"Saved visualization: {save_path}")
            count += 1
            if count >= num_samples:
                break
