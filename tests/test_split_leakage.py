"""
Unit tests for Split Integrity and Zero Leakage (Patient & Volume level).
"""

import os
import pandas as pd
import pytest


def test_split_manifests_exist():
    for name in ["train", "val", "dev"]:
        path = os.path.join("data_splits", f"{name}_manifest.csv")
        assert os.path.exists(path), f"Manifest file missing: {path}"


def test_split_counts_and_zero_leakage():
    train_df = pd.read_csv("data_splits/train_manifest.csv")
    val_df = pd.read_csv("data_splits/val_manifest.csv")
    dev_df = pd.read_csv("data_splits/dev_manifest.csv")

    # Volume counts: 134, 28, 30
    assert len(train_df) == 134, f"Train count {len(train_df)} != 134"
    assert len(val_df) == 28, f"Val count {len(val_df)} != 28"
    assert len(dev_df) == 30, f"Dev count {len(dev_df)} != 30"
    assert len(train_df) + len(val_df) + len(dev_df) == 192

    # Zero volume overlap
    s_train = set(train_df["filename"])
    s_val = set(val_df["filename"])
    s_dev = set(dev_df["filename"])

    assert len(s_train & s_val) == 0, f"VOLUME LEAKAGE between Train and Val: {s_train & s_val}"
    assert len(s_train & s_dev) == 0, f"VOLUME LEAKAGE between Train and Dev: {s_train & s_dev}"
    assert len(s_val & s_dev) == 0, f"VOLUME LEAKAGE between Val and Dev: {s_val & s_dev}"

    # Zero patient overlap
    p_train = set(train_df["patient_id"])
    p_val = set(val_df["patient_id"])
    p_dev = set(dev_df["patient_id"])

    assert len(p_train & p_val) == 0, f"PATIENT LEAKAGE between Train and Val: {p_train & p_val}"
    assert len(p_train & p_dev) == 0, f"PATIENT LEAKAGE between Train and Dev: {p_train & p_dev}"
    assert len(p_val & p_dev) == 0, f"PATIENT LEAKAGE between Val and Dev: {p_val & p_dev}"


def test_duplicate_patient_consistency():
    """
    Specifically verifies that patient '2db7ee14' (who has 2 scans) has both scans in the same split.
    """
    train_df = pd.read_csv("data_splits/train_manifest.csv")
    val_df = pd.read_csv("data_splits/val_manifest.csv")
    dev_df = pd.read_csv("data_splits/dev_manifest.csv")

    p_id = "2db7ee14"
    in_train = (train_df["patient_id"] == p_id).sum()
    in_val = (val_df["patient_id"] == p_id).sum()
    in_dev = (dev_df["patient_id"] == p_id).sum()

    assert in_train == 2 or in_val == 2 or in_dev == 2, (
        f"Multi-scan patient {p_id} was split across subsets! (train={in_train}, val={in_val}, dev={in_dev})"
    )
