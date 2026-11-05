"""
PyTorch Dataset and DataLoader builder for 2.5D Brain CT Hemorrhage Segmentation.

Key Features:
- 9-channel input: 3 consecutive slices (z-1, z, z+1) x 3 CT windows (Brain, Subdural, Bone).
- Boundary padding: Replicates edge slices for z=0 and z=Z-1.
- Target: Ground-truth mask of centre slice z mapped to the 6 target classes.
- Fast memory-efficient volume caching.
- Per-slice class indexing for minority-aware sampling.
"""

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

from src.data.transforms import CTTransform, IMAGENET_MEAN_9CH, IMAGENET_STD_9CH
from src.data.split import BHSD_TO_TARGET, TARGET_NAMES


class VolumeCache:
    """
    LRU Cache for loaded NIfTI volumes to avoid redundant disk I/O.
    """
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Tuple[np.ndarray, np.ndarray]):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value


class BrainCT25DDataset(Dataset):
    """
    2.5D Multi-Slice Brain CT Dataset.
    Each sample produces:
      image: [9, H, W] float32 tensor
      mask:  [H, W] int64 tensor with target classes in [0, 5]
    """
    def __init__(
        self,
        manifest_csv: str,
        dataset_dir: str = "Dataset",
        transform: Optional[CTTransform] = None,
        cache_capacity: int = 16,
        precache_metadata: bool = True,
        filter_empty_slices_prob: float = 0.0,
    ):
        self.dataset_dir = dataset_dir
        self.manifest_csv = manifest_csv
        self.img_dir = os.path.join(dataset_dir, "labeled", "images")
        self.gt_dir = os.path.join(dataset_dir, "labeled", "ground_truths")
        self.transform = transform or CTTransform(is_training=False)
        self.cache = VolumeCache(capacity=cache_capacity)
        
        # Load manifest
        self.manifest = pd.read_csv(manifest_csv)
        self.filenames = self.manifest["filename"].tolist()
        
        # Build slice index
        self.slice_indices: List[Dict[str, Any]] = []
        self._build_slice_index(filter_empty_slices_prob)

    def find_slice_index(self, filename: str, slice_idx: int) -> Optional[int]:
        """
        Fast lookup of a specific slice without disk I/O.
        """
        for idx, item in enumerate(self.slice_indices):
            if item["filename"] == filename and item["slice_idx"] == slice_idx:
                return idx
        return None

    def _build_slice_index(self, filter_empty_prob: float):
        """
        Creates an index of all (filename, slice_idx) pairs across the dataset partition.
        Uses precomputed slice CSV if available to avoid opening hundreds of NIfTI files.
        """
        manifest_dir = os.path.dirname(self.manifest_csv) if hasattr(self, "manifest_csv") else "data_splits"
        manifest_base = os.path.basename(self.manifest_csv) if hasattr(self, "manifest_csv") else "train_manifest.csv"
        split_name = manifest_base.replace("_manifest.csv", "")
        slice_cache_csv = os.path.join(manifest_dir, f"{split_name}_slices.csv")

        if os.path.exists(slice_cache_csv):
            slices_df = pd.read_csv(slice_cache_csv)
            for _, row in slices_df.iterrows():
                has_fg = bool(row["has_foreground"])
                if not has_fg and filter_empty_prob > 0:
                    if np.random.rand() < filter_empty_prob:
                        continue
                raw_classes_str = str(row["target_classes"])
                t_classes = [int(c) for c in raw_classes_str.split(",") if c.strip().isdigit()]
                self.slice_indices.append({
                    "filename": str(row["filename"]),
                    "slice_idx": int(row["slice_idx"]),
                    "num_slices": int(row["num_slices"]),
                    "target_classes": t_classes,
                    "has_foreground": has_fg,
                    "has_edh": bool(row["has_edh"]),
                    "has_iph": bool(row["has_iph"]),
                    "has_sah": bool(row["has_sah"]),
                    "has_sdh": bool(row["has_sdh"]),
                    "has_ivh": bool(row["has_ivh"]),
                })
            return

        for _, row in self.manifest.iterrows():
            fname = row["filename"]
            num_slices = int(row["num_slices"])
            gt_path = os.path.join(self.gt_dir, fname)
            
            # Load GT to determine slice-level labels
            gt_nii = nib.load(gt_path)
            gt_data = np.asanyarray(gt_nii.dataobj)
            
            for z in range(num_slices):
                slice_gt = gt_data[:, :, z]
                unique_raw = [int(c) for c in np.unique(slice_gt)]
                unique_target = [BHSD_TO_TARGET[c] for c in unique_raw]
                has_fg = any(c > 0 for c in unique_target)
                
                # Optional skip of some background-only slices during training if requested
                if not has_fg and filter_empty_prob > 0:
                    if np.random.rand() < filter_empty_prob:
                        continue
                        
                self.slice_indices.append({
                    "filename": fname,
                    "slice_idx": z,
                    "num_slices": num_slices,
                    "target_classes": unique_target,
                    "has_foreground": has_fg,
                    "has_edh": 3 in unique_target,
                    "has_iph": 1 in unique_target,
                    "has_sah": 2 in unique_target,
                    "has_sdh": 4 in unique_target,
                    "has_ivh": 5 in unique_target,
                })

            gt_nii.uncache()
            del gt_nii

    def __len__(self) -> int:
        return len(self.slice_indices)

    def _load_volume(self, fname: str) -> Tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(fname)
        if cached is not None:
            return cached

        img_path = os.path.join(self.img_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)

        # Store as int16 (HU values [-3024, 3072] fit cleanly) directly from dataobj
        img_int16 = np.asanyarray(img_nii.dataobj).astype(np.int16)
        gt_data = np.asanyarray(gt_nii.dataobj)

        # Map raw BHSD classes to target classes
        target_gt = np.zeros(gt_data.shape, dtype=np.uint8)
        for raw_c, target_c in BHSD_TO_TARGET.items():
            if target_c != 0:
                target_gt[gt_data == raw_c] = target_c

        img_nii.uncache()
        gt_nii.uncache()
        del img_nii, gt_nii

        self.cache.put(fname, (img_int16, target_gt))
        return img_int16, target_gt

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta = self.slice_indices[idx]
        fname = meta["filename"]
        z = meta["slice_idx"]
        num_slices = meta["num_slices"]

        img_vol, gt_vol = self._load_volume(fname)

        # 2.5D consecutive slice indices with edge padding (replicate boundaries)
        z_prev = max(0, z - 1)
        z_curr = z
        z_next = min(num_slices - 1, z + 1)

        # Extract [3, H, W] slices in HU
        # NIfTI data is typically (H, W, Z)
        s_prev = img_vol[:, :, z_prev]
        s_curr = img_vol[:, :, z_curr]
        s_next = img_vol[:, :, z_next]
        slices_hu = np.stack([s_prev, s_curr, s_next], axis=0).astype(np.float32)

        # Centre slice mask [H, W]
        mask = gt_vol[:, :, z_curr].astype(np.int64)

        # Apply transformations (windowing, augmentations, normalization)
        norm_9ch, processed_mask = self.transform(slices_hu, mask)

        t_img = torch.from_numpy(norm_9ch).float()      # [9, H, W]
        t_mask = torch.from_numpy(processed_mask).long() # [H, W]

        has_classes = torch.zeros(6, dtype=torch.bool)
        target_classes = meta.get("target_classes", [])
        if isinstance(target_classes, str):
            target_classes = [int(c) for c in target_classes.split(",") if c.strip().isdigit()]
        for c in target_classes:
            if 0 <= c < 6:
                has_classes[c] = True

        sample_info = {
            "filename": fname,
            "slice_idx": z,
            "has_classes": has_classes,
        }

        return t_img, t_mask, sample_info
