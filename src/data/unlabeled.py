"""
Unlabeled CT Dataset Loader and Discovery Module for Active Learning.

Supports:
- Auto-discovery of unlabeled brain CT scans from 'anybleed' and 'nobleed' folders.
- Both 3D NIfTI volumes (.nii, .nii.gz) and 2D CT images (.png, .jpg, .jpeg, .tiff, .dcm).
- 2.5D multi-window slice extraction: 3 consecutive slices (z-1, z, z+1) x 3 HU windows = 9 channels.
- Fast HU pre-filtering (skips slices with max brain HU < 60 before scoring).
- High-performance LRU volume caching to minimize disk I/O.
"""

import os
import glob
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import nibabel as nib

from src.data.transforms import (
    apply_3_windows,
    build_25d_input,
    normalize_channels,
    WINDOW_SPECS,
    IMAGENET_MEAN_9CH,
    IMAGENET_STD_9CH,
)


def discover_unlabeled_scans(
    unlabeled_dir: Optional[str] = None,
    dataset_dir: str = "Dataset",
    max_scans: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Discovers unlabeled CT scans in 'anybleed' and 'nobleed' folders.
    Checks:
    1. Explicit unlabeled_dir if provided.
    2. os.path.join(dataset_dir, "unlabeled")
    3. dataset_dir (if 'anybleed' or 'nobleed' are direct subfolders)
    
    Args:
        unlabeled_dir: Optional explicit path.
        dataset_dir: Root dataset folder.
        max_scans: Optional max total scans to return (balanced across anybleed/nobleed).
    
    Returns:
        List of dicts with:
        - path: absolute/relative path to scan
        - filename: file name
        - study_id: patient or study ID
        - category: 'anybleed' or 'nobleed'
        - file_type: 'nifti' or 'image'
    """
    candidate_roots = []
    if unlabeled_dir and os.path.exists(unlabeled_dir):
        candidate_roots.append(unlabeled_dir)
    
    unlabeled_sub = os.path.join(dataset_dir, "unlabeled")
    if os.path.exists(unlabeled_sub):
        candidate_roots.append(unlabeled_sub)
        
    if os.path.exists(dataset_dir):
        candidate_roots.append(dataset_dir)

    categories = ["anybleed", "nobleed"]
    valid_exts = {
        "nifti": [".nii.gz", ".nii"],
        "image": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"],
    }

    discovered = []
    seen_paths = set()

    for root in candidate_roots:
        by_cat: Dict[str, List[Dict[str, Any]]] = {c: [] for c in categories}
        for cat in categories:
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir):
                continue

            for dirpath, _, filenames in os.walk(cat_dir):
                for f in sorted(filenames):
                    f_lower = f.lower()
                    file_path = os.path.join(dirpath, f)
                    if file_path in seen_paths:
                        continue

                    # Determine file type
                    f_type = None
                    if any(f_lower.endswith(ext) for ext in valid_exts["nifti"]):
                        f_type = "nifti"
                    elif any(f_lower.endswith(ext) for ext in valid_exts["image"]):
                        f_type = "image"

                    if f_type is not None:
                        # Extract study ID from filename
                        base_name = f
                        for ext in [".nii.gz", ".nii", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm"]:
                            if base_name.lower().endswith(ext):
                                base_name = base_name[:-len(ext)]
                                break

                        seen_paths.add(file_path)
                        item = {
                            "path": file_path,
                            "filename": f,
                            "study_id": base_name,
                            "category": cat,
                            "file_type": f_type,
                            "rel_dir": os.path.relpath(dirpath, root),
                        }
                        by_cat[cat].append(item)

        # Balance across anybleed and nobleed if max_scans is requested
        total_found = sum(len(items) for items in by_cat.values())
        if total_found > 0:
            if max_scans is not None and max_scans < total_found:
                per_cat = max(1, max_scans // 2)
                for cat in categories:
                    discovered.extend(by_cat[cat][:per_cat])
                # Top-up if shortfall
                if len(discovered) < max_scans:
                    for cat in categories:
                        rem = max_scans - len(discovered)
                        discovered.extend(by_cat[cat][per_cat:per_cat + rem])
            else:
                for cat in categories:
                    discovered.extend(by_cat[cat])
            break

    return discovered


class UnlabeledVolumeCache:
    """
    LRU Cache for unlabelled 3D CT volumes to avoid redundant disk I/O.
    """
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: np.ndarray):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value


class UnlabeledCTDataset(Dataset):
    """
    PyTorch Dataset for Unlabeled CT Scans (anybleed and nobleed).
    Extracts 2.5D multi-window inputs (9 channels) matching DeepLabV3+ input specs.
    """
    def __init__(
        self,
        unlabeled_dir: Optional[str] = None,
        dataset_dir: str = "Dataset",
        cache_capacity: int = 8,
        target_size: Optional[Tuple[int, int]] = (512, 512),
        max_scans: Optional[int] = None,
    ):
        self.dataset_dir = dataset_dir
        self.unlabeled_dir = unlabeled_dir
        self.target_size = target_size
        self.cache = UnlabeledVolumeCache(capacity=cache_capacity)

        # Discover all scans
        self.scans = discover_unlabeled_scans(
            unlabeled_dir=unlabeled_dir,
            dataset_dir=dataset_dir,
            max_scans=max_scans,
        )
        
        # Build slice index
        self.slice_indices: List[Dict[str, Any]] = []
        self._build_slice_index()

    def find_slice_index(self, filename: str, slice_idx: int) -> Optional[int]:
        """
        Fast lookup of a specific slice without disk I/O.
        """
        for idx, item in enumerate(self.slice_indices):
            if item["filename"] == filename and item["slice_idx"] == slice_idx:
                return idx
        return None

    def _build_slice_index(self):
        """
        Catalogs all available slices from discovered scans.
        """
        for scan in self.scans:
            f_type = scan["file_type"]
            path = scan["path"]

            if f_type == "nifti":
                try:
                    # Open header only to obtain shape without full array read
                    nii = nib.load(path)
                    shape = nii.header.get_data_shape()
                    # Identify slice axis (typically axis 2, or axis 0 if (Z, H, W))
                    if len(shape) >= 3:
                        if shape[0] < shape[2] and shape[0] <= 120:
                            num_slices = int(shape[0])
                            slice_axis = 0
                        else:
                            num_slices = int(shape[2])
                            slice_axis = 2
                    else:
                        num_slices = 1
                        slice_axis = -1

                    for z in range(num_slices):
                        self.slice_indices.append({
                            "scan": scan,
                            "slice_idx": z,
                            "num_slices": num_slices,
                            "slice_axis": slice_axis,
                            "filename": scan["filename"],
                            "study_id": scan["study_id"],
                            "category": scan["category"],
                            "file_type": "nifti",
                        })
                except Exception as e:
                    print(f"[WARNING] Could not read NIfTI header for {path}: {e}")

            elif f_type == "image":
                self.slice_indices.append({
                    "scan": scan,
                    "slice_idx": 0,
                    "num_slices": 1,
                    "slice_axis": -1,
                    "filename": scan["filename"],
                    "study_id": scan["study_id"],
                    "category": scan["category"],
                    "file_type": "image",
                })

    def __len__(self) -> int:
        return len(self.slice_indices)

    def _load_nifti_volume(self, path: str, slice_axis: int) -> np.ndarray:
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        nii = nib.load(path)
        data = np.asanyarray(nii.dataobj).astype(np.float32)
        
        # Orient so slices are along axis 2 (H, W, Z)
        if slice_axis == 0 and data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        nii.uncache()
        del nii
        self.cache.put(path, data)
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        item = self.slice_indices[idx]
        scan = item["scan"]
        f_type = item["file_type"]
        path = scan["path"]
        z = item["slice_idx"]
        num_slices = item["num_slices"]

        if f_type == "nifti":
            vol = self._load_nifti_volume(path, item["slice_axis"])
            h, w, total_z = vol.shape[:3]

            # 2.5D slice indices with edge clamping
            z_prev = max(0, z - 1)
            z_curr = min(total_z - 1, max(0, z))
            z_next = min(total_z - 1, z + 1)

            s_prev = vol[:, :, z_prev]
            s_curr = vol[:, :, z_curr]
            s_next = vol[:, :, z_next]

            # Fast HU pre-filter check on center slice
            # Brain window: HU range [0, 80]
            brain_hu_vals = s_curr[(s_curr >= 0) & (s_curr <= 100)]
            max_brain_hu = float(np.max(brain_hu_vals)) if len(brain_hu_vals) > 0 else float(np.max(s_curr))
            passes_hu_prefilter = bool(max_brain_hu >= 60.0)

            # Assemble 9-channel 2.5D input (3 slices x 3 windows)
            norm_9ch = build_25d_input(s_prev, s_curr, s_next)  # [9, H, W] in [0, 1]
            norm_9ch = normalize_channels(norm_9ch)              # ImageNet normalized

        elif path.lower().endswith(".dcm"):
            # DICOM CT slice
            try:
                import pydicom
                dcm = pydicom.dcmread(path)
                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
                hu_slice = dcm.pixel_array.astype(np.float32) * slope + intercept
                brain_hu_vals = hu_slice[(hu_slice >= 0) & (hu_slice <= 100)]
                max_brain_hu = float(np.max(brain_hu_vals)) if len(brain_hu_vals) > 0 else float(np.max(hu_slice))
                passes_hu_prefilter = bool(max_brain_hu >= 60.0)
                norm_9ch = build_25d_input(hu_slice, hu_slice, hu_slice)
                norm_9ch = normalize_channels(norm_9ch)
            except Exception as e:
                print(f"[WARNING] Could not read DICOM {path}: {e}")
                norm_9ch = np.zeros((9, 512, 512), dtype=np.float32)
                max_brain_hu = 0.0
                passes_hu_prefilter = False

        else:
            # 2D Image slice (png, jpg, tiff)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)

            if img.ndim == 2:
                # Grayscale image
                norm_img = img.astype(np.float32)
                if norm_img.max() > 1.0:
                    norm_img = norm_img / 255.0
                # Replicate to 9 channels
                norm_9ch = np.repeat(norm_img[None, :, :], 9, axis=0)
                norm_9ch = normalize_channels(norm_9ch)
                max_brain_hu = float(img.max())
                passes_hu_prefilter = True
            elif img.ndim == 3 and img.shape[2] == 3:
                # 3-channel (e.g. RGB or 3 precomputed CT windows)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                rgb_ch = np.transpose(rgb, (2, 0, 1))  # [3, H, W]
                # Replicate 3 times for 2.5D consecutive slices (z-1, z, z+1)
                norm_9ch = np.repeat(rgb_ch, 3, axis=0)  # [9, H, W]
                norm_9ch = normalize_channels(norm_9ch)
                max_brain_hu = float(img.max())
                passes_hu_prefilter = True
            else:
                norm_9ch = np.zeros((9, 512, 512), dtype=np.float32)
                max_brain_hu = 0.0
                passes_hu_prefilter = False

        # Optional resize
        if self.target_size is not None and (norm_9ch.shape[1] != self.target_size[0] or norm_9ch.shape[2] != self.target_size[1]):
            resized_channels = []
            for c in range(norm_9ch.shape[0]):
                resized = cv2.resize(norm_9ch[c], (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized)
            norm_9ch = np.stack(resized_channels, axis=0)

        tensor_img = torch.from_numpy(norm_9ch).float()

        meta = {
            "filename": item["filename"],
            "study_id": item["study_id"],
            "slice_idx": int(z),
            "num_slices": int(num_slices),
            "category": item["category"],
            "file_type": item["file_type"],
            "max_brain_hu": float(max_brain_hu),
            "passes_hu_prefilter": passes_hu_prefilter,
        }

        return tensor_img, meta
