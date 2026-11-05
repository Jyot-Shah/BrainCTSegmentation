"""
Class-aware Slice Sampler for Handling Severe Minority Imbalance.

Boosts the sampling frequency of slices containing rare hemorrhage types
(especially SDH, SAH, and EDH) while preserving a realistic mixture of
common hemorrhage and background slices.
"""

from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def create_class_aware_sampler(
    slice_indices: List[Dict[str, Any]],
    weight_edh: float = 30.0,
    weight_sdh: float = 8.0,
    weight_sah: float = 6.0,
    weight_iph: float = 2.0,
    weight_ivh: float = 2.5,
    weight_bg: float = 0.4,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Computes sample weights per slice based on highest-priority hemorrhage class present.
    Priority order: EDH > SDH > SAH > IVH > IPH > BG.
    
    Returns:
        PyTorch WeightedRandomSampler
    """
    weights = []
    
    for meta in slice_indices:
        target_classes = set(meta.get("target_classes", []))
        
        # Target classes:
        # 1: IPH, 2: SAH, 3: EDH, 4: SDH, 5: IVH, 0: BG
        if 3 in target_classes:
            w = weight_edh
        elif 4 in target_classes:
            w = weight_sdh
        elif 2 in target_classes:
            w = weight_sah
        elif 5 in target_classes:
            w = weight_ivh
        elif 1 in target_classes:
            w = weight_iph
        else:
            w = weight_bg
            
        weights.append(w)
        
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    num_samples = len(weights)
    
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
    )
    return sampler
