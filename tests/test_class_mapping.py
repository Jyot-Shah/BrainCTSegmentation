"""
Unit tests for Class Mapping, Target Nomenclature, and Mask Dimensions.
"""

import os
import nibabel as nib
import numpy as np
import pytest
from src.data.split import BHSD_TO_TARGET, TARGET_NAMES, RAW_BHSD_CLASSES


def test_target_class_definitions():
    """
    Verifies that the target classes match the required 6 classes in the exact specified order:
    0: Background
    1: IPH (intraparenchymal)
    2: SAH (subarachnoid)
    3: EDH (epidural)
    4: SDH (subdural)
    5: IVH (intraventricular)
    """
    assert len(TARGET_NAMES) == 6
    assert TARGET_NAMES[0] == "Background"
    assert TARGET_NAMES[1] == "IPH"
    assert TARGET_NAMES[2] == "SAH"
    assert TARGET_NAMES[3] == "EDH"
    assert TARGET_NAMES[4] == "SDH"
    assert TARGET_NAMES[5] == "IVH"


def test_raw_to_target_bijection():
    """
    Verifies that each of the 6 raw BHSD classes maps to a unique target integer in [0, 5].
    """
    assert len(BHSD_TO_TARGET) == 6
    mapped_targets = list(BHSD_TO_TARGET.values())
    assert sorted(mapped_targets) == [0, 1, 2, 3, 4, 5]

    # Check semantic correctness
    # Raw BHSD: 0=bg, 1=edh, 2=iph, 3=ivh, 4=sah, 5=sdh
    assert BHSD_TO_TARGET[0] == 0  # Background -> Background
    assert BHSD_TO_TARGET[1] == 3  # EDH -> EDH (class 3)
    assert BHSD_TO_TARGET[2] == 1  # IPH -> IPH (class 1)
    assert BHSD_TO_TARGET[3] == 5  # IVH -> IVH (class 5)
    assert BHSD_TO_TARGET[4] == 2  # SAH -> SAH (class 2)
    assert BHSD_TO_TARGET[5] == 4  # SDH -> SDH (class 4)


def test_mask_image_spatial_dimensions():
    """
    Checks that sample NIfTI image and mask files have perfectly identical spatial dimensions.
    """
    img_dir = "Dataset/labeled/images"
    gt_dir = "Dataset/labeled/ground_truths"
    files = sorted(os.listdir(img_dir))[:3]

    for f in files:
        img_nii = nib.load(os.path.join(img_dir, f))
        gt_nii = nib.load(os.path.join(gt_dir, f))
        assert img_nii.shape == gt_nii.shape, (
            f"Shape mismatch in {f}: {img_nii.shape} vs {gt_nii.shape}"
        )
