"""
Split generator for BHSD Brain CT Hemorrhage Dataset.

Ensures:
1. Exactly 134 training, 28 validation, and 30 development holdout volumes (total 192).
2. Zero volume overlap and zero patient overlap across splits (guards against multi-scan patients like ID_2db7ee14).
3. Stratified allocation so all 5 hemorrhage classes (especially rare EDH, SDH, SAH) are represented in train, val, and dev.
4. Generates reproducible CSV manifests with metadata.
"""

import os
import re
import json
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import nibabel as nib

# Raw BHSD class mapping:
# 0: Background, 1: EDH, 2: IPH, 3: IVH, 4: SAH, 5: SDH
RAW_BHSD_CLASSES = {
    0: "background",
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}

# Target 6-class specification:
# 0: Background, 1: IPH, 2: SAH, 3: EDH, 4: SDH, 5: IVH
BHSD_TO_TARGET = {
    0: 0,  # Background
    1: 3,  # EDH
    2: 1,  # IPH
    3: 5,  # IVH
    4: 2,  # SAH
    5: 4,  # SDH
}

TARGET_NAMES = ["Background", "IPH", "SAH", "EDH", "SDH", "IVH"]


def extract_patient_id(filename: str) -> str:
    """Extract patient identifier from filename like ID_0237f3c9_ID_40015688b9.nii.gz."""
    parts = filename.replace(".nii.gz", "").split("_")
    if len(parts) >= 2 and parts[0] == "ID":
        return parts[1]
    return parts[0]


def extract_study_id(filename: str) -> str:
    """Extract study identifier from filename."""
    parts = filename.replace(".nii.gz", "").split("_")
    if len(parts) >= 4 and parts[2] == "ID":
        return parts[3]
    return parts[-1]


def inspect_volumes_metadata(dataset_dir: str):
    """
    Scans all volumes and extracts patient ID, slice count, shape, and class presence.
    """
    img_dir = os.path.join(dataset_dir, "labeled", "images")
    gt_dir = os.path.join(dataset_dir, "labeled", "ground_truths")
    
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".nii.gz")])
    metadata = []
    
    for f in files:
        p_id = extract_patient_id(f)
        s_id = extract_study_id(f)
        gt_path = os.path.join(gt_dir, f)
        
        # Load mask to get shape and unique classes
        gt_nii = nib.load(gt_path)
        gt_arr = np.asanyarray(gt_nii.dataobj)
        shape = list(gt_arr.shape)
        unique_raw = [int(c) for c in np.unique(gt_arr)]
        unique_target = [BHSD_TO_TARGET[c] for c in unique_raw]
        
        has_edh = 1 in unique_raw
        has_iph = 2 in unique_raw
        has_ivh = 3 in unique_raw
        has_sah = 4 in unique_raw
        has_sdh = 5 in unique_raw
        
        metadata.append({
            "filename": f,
            "patient_id": p_id,
            "study_id": s_id,
            "num_slices": shape[2],
            "height": shape[0],
            "width": shape[1],
            "raw_classes": unique_raw,
            "target_classes": unique_target,
            "has_edh": has_edh,
            "has_iph": has_iph,
            "has_ivh": has_ivh,
            "has_sah": has_sah,
            "has_sdh": has_sdh,
            "any_hemorrhage": len(unique_raw) > 1,
        })
        
    return pd.DataFrame(metadata)


def create_stratified_split(
    df: pd.DataFrame,
    n_train: int = 134,
    n_val: int = 28,
    n_dev: int = 30,
    seed: int = 42,
) -> dict:
    """
    Create patient-level stratified split: 134 train, 28 val, 30 dev holdout.
    Enforces that patients with multiple scans are NEVER split across subsets.
    """
    total = len(df)
    assert total == n_train + n_val + n_dev, (
        f"Sum of splits {n_train}+{n_val}+{n_dev} must equal total volumes {total}!"
    )
    
    rng = np.random.RandomState(seed)
    
    # Group by patient_id
    patient_groups = df.groupby("patient_id")
    patient_info = []
    
    for p_id, group in patient_groups:
        files = group["filename"].tolist()
        has_edh = group["has_edh"].any()
        has_iph = group["has_iph"].any()
        has_ivh = group["has_ivh"].any()
        has_sah = group["has_sah"].any()
        has_sdh = group["has_sdh"].any()
        
        # Priority for stratification: rarest class first (EDH > SDH > IVH > SAH > IPH)
        if has_edh:
            strat_key = "EDH"
        elif has_sdh:
            strat_key = "SDH"
        elif has_ivh:
            strat_key = "IVH"
        elif has_sah:
            strat_key = "SAH"
        elif has_iph:
            strat_key = "IPH"
        else:
            strat_key = "None"
            
        patient_info.append({
            "patient_id": p_id,
            "filenames": files,
            "num_volumes": len(files),
            "strat_key": strat_key,
            "has_edh": has_edh,
            "has_sdh": has_sdh,
            "has_ivh": has_ivh,
            "has_sah": has_sah,
            "has_iph": has_iph,
        })
        
    # Shuffle patients deterministically
    rng.shuffle(patient_info)
    
    # Stratified bucket distribution
    buckets = defaultdict(list)
    for p in patient_info:
        buckets[p["strat_key"]].append(p)
        
    train_patients, val_patients, dev_patients = [], [], []
    train_vol_count, val_vol_count, dev_vol_count = 0, 0, 0
    
    # Sort keys by rarity to distribute rare classes first
    key_order = ["EDH", "SDH", "IVH", "SAH", "IPH", "None"]
    
    for key in key_order:
        items = buckets[key]
        for p in items:
            v_count = p["num_volumes"]
            
            # Target ratios: 134/192 ~ 0.698, 28/192 ~ 0.146, 30/192 ~ 0.156
            # Check who needs volume most without exceeding max
            train_deficit = (n_train - train_vol_count)
            val_deficit = (n_val - val_vol_count)
            dev_deficit = (n_dev - dev_vol_count)
            
            # Fraction filled
            train_frac = train_vol_count / n_train
            val_frac = val_vol_count / n_val
            dev_frac = dev_vol_count / n_dev
            
            if val_frac <= dev_frac and val_deficit >= v_count and val_vol_count < n_val:
                val_patients.append(p)
                val_vol_count += v_count
            elif dev_frac <= train_frac and dev_deficit >= v_count and dev_vol_count < n_dev:
                dev_patients.append(p)
                dev_vol_count += v_count
            elif train_deficit >= v_count:
                train_patients.append(p)
                train_vol_count += v_count
            elif val_deficit >= v_count:
                val_patients.append(p)
                val_vol_count += v_count
            elif dev_deficit >= v_count:
                dev_patients.append(p)
                dev_vol_count += v_count
            else:
                train_patients.append(p)
                train_vol_count += v_count

    # Flatten filenames
    train_files = [f for p in train_patients for f in p["filenames"]]
    val_files = [f for p in val_patients for f in p["filenames"]]
    dev_files = [f for p in dev_patients for f in p["filenames"]]
    
    # Assert exact counts
    assert len(train_files) == n_train, f"Train count {len(train_files)} != {n_train}"
    assert len(val_files) == n_val, f"Val count {len(val_files)} != {n_val}"
    assert len(dev_files) == n_dev, f"Dev count {len(dev_files)} != {n_dev}"
    
    # Assert zero volume leakage
    s_train, s_val, s_dev = set(train_files), set(val_files), set(dev_files)
    assert len(s_train & s_val) == 0, f"LEAKAGE: Train and Val overlap on {s_train & s_val}"
    assert len(s_train & s_dev) == 0, f"LEAKAGE: Train and Dev overlap on {s_train & s_dev}"
    assert len(s_val & s_dev) == 0, f"LEAKAGE: Val and Dev overlap on {s_val & s_dev}"
    
    # Assert zero patient leakage
    p_train = {extract_patient_id(f) for f in train_files}
    p_val = {extract_patient_id(f) for f in val_files}
    p_dev = {extract_patient_id(f) for f in dev_files}
    assert len(p_train & p_val) == 0, f"PATIENT LEAKAGE: Train and Val overlap on {p_train & p_val}"
    assert len(p_train & p_dev) == 0, f"PATIENT LEAKAGE: Train and Dev overlap on {p_train & p_dev}"
    assert len(p_val & p_dev) == 0, f"PATIENT LEAKAGE: Val and Dev overlap on {p_val & p_dev}"
    
    return {
        "train": train_files,
        "val": val_files,
        "dev": dev_files,
    }


def generate_and_save_splits(
    dataset_dir: str = "Dataset",
    output_dir: str = "data_splits",
    seed: int = 42,
):
    """
    Main utility to inspect dataset, create split, and save manifests.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = inspect_volumes_metadata(dataset_dir)
    splits = create_stratified_split(df, n_train=134, n_val=28, n_dev=30, seed=seed)
    
    manifest_paths = {}
    for split_name, flist in splits.items():
        sub_df = df[df["filename"].isin(flist)].copy()
        sub_df["split"] = split_name
        csv_path = os.path.join(output_dir, f"{split_name}_manifest.csv")
        sub_df.to_csv(csv_path, index=False)
        manifest_paths[split_name] = csv_path
        
        # Log class representation in this split
        print(f"\n--- {split_name.upper()} SPLIT (Volumes={len(flist)}) ---")
        for cls_name, col in [("EDH", "has_edh"), ("IPH", "has_iph"), ("IVH", "has_ivh"), 
                              ("SAH", "has_sah"), ("SDH", "has_sdh")]:
            count = sub_df[col].sum()
            print(f"  {cls_name}: {count}/{len(flist)} volumes ({count/len(flist)*100:.1f}%)")
            
    # Also save full combined manifest
    full_manifest = df.copy()
    split_map = {}
    for s_name, flist in splits.items():
        for f in flist:
            split_map[f] = s_name
    full_manifest["split"] = full_manifest["filename"].map(split_map)
    full_manifest.to_csv(os.path.join(output_dir, "full_manifest.csv"), index=False)
    
    print("\n[SUCCESS] Split generation complete with 0 patient/volume leakage!")
    return manifest_paths


if __name__ == "__main__":
    generate_and_save_splits()
