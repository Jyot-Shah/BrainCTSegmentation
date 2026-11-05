"""
CT Windowing, Normalization, and Augmentation Pipeline for 2.5D Brain CT.

Retains the exact multi-window CT strategy:
- Brain window:    WL=40,  WW=80   -> HU range [0, 80]
- Subdural window: WL=75,  WW=215  -> HU range [-32.5, 182.5]
- Bone window:     WL=400, WW=1800 -> HU range [-500, 1300]

9-channel 2.5D input:
- 3 consecutive slices (z-1, z, z+1) x 3 windows each = 9 channels.
- Normalization: ImageNet mean/std per channel triplet.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# CT Window Configurations: (WL, WW)
WINDOW_SPECS = {
    "brain": (40.0, 80.0),
    "subdural": (75.0, 215.0),
    "bone": (400.0, 1800.0),
}

# ImageNet statistics for normalization
IMAGENET_MEAN_3CH = [0.485, 0.456, 0.406]
IMAGENET_STD_3CH = [0.229, 0.224, 0.225]

IMAGENET_MEAN_9CH = IMAGENET_MEAN_3CH * 3
IMAGENET_STD_9CH = IMAGENET_STD_3CH * 3


def apply_window(img_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """
    Applies a single CT window to an image in HU units.
    Returns normalized values in range [0.0, 1.0].
    
    Formula:
        lower = wl - ww / 2
        upper = wl + ww / 2
        windowed = clip(img_hu, lower, upper)
        normalized = (windowed - lower) / (upper - lower)
    """
    lower = wl - (ww / 2.0)
    upper = wl + (ww / 2.0)
    windowed = np.clip(img_hu, lower, upper)
    normalized = (windowed - lower) / (upper - lower)
    return normalized.astype(np.float32)


def apply_3_windows(img_hu: np.ndarray) -> np.ndarray:
    """
    Applies Brain, Subdural, and Bone windows to a 2D slice [H, W].
    Returns [3, H, W] array with values in [0.0, 1.0].
    """
    w_brain = apply_window(img_hu, *WINDOW_SPECS["brain"])
    w_subdural = apply_window(img_hu, *WINDOW_SPECS["subdural"])
    w_bone = apply_window(img_hu, *WINDOW_SPECS["bone"])
    return np.stack([w_brain, w_subdural, w_bone], axis=0)


def build_25d_input(
    slice_prev: np.ndarray,
    slice_curr: np.ndarray,
    slice_next: np.ndarray,
) -> np.ndarray:
    """
    Assembles 9-channel 2.5D input from three consecutive CT slices in HU.
    Channels 0-2: Brain, Subdural, Bone for slice_prev
    Channels 3-5: Brain, Subdural, Bone for slice_curr (centre)
    Channels 6-8: Brain, Subdural, Bone for slice_next
    Returns array of shape [9, H, W] in range [0.0, 1.0].
    """
    stack_prev = apply_3_windows(slice_prev)
    stack_curr = apply_3_windows(slice_curr)
    stack_next = apply_3_windows(slice_next)
    return np.concatenate([stack_prev, stack_curr, stack_next], axis=0)


def normalize_channels(
    arr: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Normalizes multi-channel array [C, H, W] using channel-specific mean and std.
    """
    c = arr.shape[0]
    if mean is None:
        mean = IMAGENET_MEAN_9CH if c == 9 else IMAGENET_MEAN_3CH
    if std is None:
        std = IMAGENET_STD_9CH if c == 9 else IMAGENET_STD_3CH
        
    norm_arr = np.empty_like(arr, dtype=np.float32)
    for i in range(c):
        norm_arr[i] = (arr[i] - mean[i]) / std[i]
    return norm_arr


class CTTransform:
    """
    Clinically-motivated spatial & conservative intensity transformations for 2.5D brain CT.
    Conserves HU windowing integrity by bounding intensity shifts within +/-30 HU.
    """
    def __init__(
        self,
        is_training: bool = True,
        hflip_prob: float = 0.5,
        rotate_deg: float = 15.0,
        intensity_jitter_hu: float = 20.0,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.is_training = is_training
        self.hflip_prob = hflip_prob
        self.rotate_deg = rotate_deg
        self.intensity_jitter_hu = min(intensity_jitter_hu, 30.0)  # Capped at 30 HU
        self.target_size = target_size

    def __call__(
        self,
        slices_hu: np.ndarray,  # shape [3, H, W] in HU units (prev, curr, next)
        mask: np.ndarray,       # shape [H, W] integer class labels
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies conservative transformations.
        Returns:
            processed_9ch: [9, H, W] float32 normalized image tensor
            processed_mask: [H, W] int64 class mask
        """
        slices_hu = slices_hu.copy()
        mask = mask.copy()

        if self.is_training:
            # 1. Conservative HU intensity jitter (+/- 20-30 HU)
            if self.intensity_jitter_hu > 0 and np.random.rand() > 0.5:
                delta = np.random.uniform(-self.intensity_jitter_hu, self.intensity_jitter_hu)
                slices_hu = slices_hu + delta

            # 2. Random horizontal flip (brain CT is anatomically symmetric across midline)
            if np.random.rand() < self.hflip_prob:
                slices_hu = np.flip(slices_hu, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            # 3. Modest rotation (+/- 15 degrees) via fast C++ OpenCV warpAffine
            if self.rotate_deg > 0 and np.random.rand() > 0.5:
                angle = float(np.random.uniform(-self.rotate_deg, self.rotate_deg))
                h, w = slices_hu.shape[1], slices_hu.shape[2]
                center = (w / 2.0, h / 2.0)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                for i in range(3):
                    slices_hu[i] = cv2.warpAffine(slices_hu[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask.astype(np.int32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT).astype(np.int64)

        # Build 9-channel windowed representation
        stacked_9ch = build_25d_input(slices_hu[0], slices_hu[1], slices_hu[2])
        
        # Spatial resize if target_size is specified
        if self.target_size is not None:
            t_9ch = torch.from_numpy(stacked_9ch).unsqueeze(0)
            t_m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            
            t_9ch = F.interpolate(t_9ch, size=self.target_size, mode="bilinear", align_corners=False)
            t_m = F.interpolate(t_m, size=self.target_size, mode="nearest")
            
            stacked_9ch = t_9ch.squeeze(0).numpy()
            mask = t_m.squeeze(0).squeeze(0).long().numpy()

        # ImageNet channel-wise normalization
        norm_9ch = normalize_channels(stacked_9ch)

        return norm_9ch, mask.astype(np.int64)
