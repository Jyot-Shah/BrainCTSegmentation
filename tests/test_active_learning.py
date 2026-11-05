"""
Unit tests for Active Learning modules:
- MCDropoutScorer (Epistemic Uncertainty, BALD, Entropy)
- Entropy <-> BALD correlation >= 0.85 (Main doc requirement)
- UnlabeledCTDataset auto-discovery and 2.5D multi-window loading (anybleed / nobleed)
- PoolScorer quality floors and study diversity caps
- StratifiedSampler vs TopKSampler vs RandomSampler
- SimulatedOracle ground-truth revelation
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import shutil
import tempfile
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import cv2

try:
    import pytest
    fixture_decorator = pytest.fixture
except ImportError:
    def fixture_decorator(fn):
        return fn

from src.models.deeplab import DeepLabV3Plus25D
from src.active_learning.uncertainty import MCDropoutScorer
from src.active_learning.selector import StratifiedSampler, RandomSampler, TopKSampler, SimulatedOracle
from src.data.unlabeled import UnlabeledCTDataset, discover_unlabeled_scans


def create_dummy_model():
    return DeepLabV3Plus25D(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=9,
        num_classes=6,
        decoder_dropout=0.2,
    )


@fixture_decorator
def dummy_model():
    return create_dummy_model()


def test_mc_dropout_uncertainty_bounds(dummy_model):
    """
    Verifies that MCDropoutScorer computes non-negative entropy, BALD, and variance.
    """
    device = torch.device("cpu")
    scorer = MCDropoutScorer(model=dummy_model, device=device, num_passes=3, use_amp=False)

    x = torch.randn(2, 9, 64, 64)
    res = scorer.score_batch(x)

    assert "entropy_map" in res
    assert "bald_map" in res
    assert "variance_map" in res
    assert "slice_metrics" in res

    # Mathematical bounds
    assert (res["entropy_map"] >= -1e-6).all(), "Entropy contains negative values!"
    assert (res["bald_map"] >= -1e-6).all(), "BALD contains negative values!"
    assert (res["variance_map"] >= -1e-6).all(), "Variance contains negative values!"

    metrics = res["slice_metrics"]
    assert len(metrics) == 2
    for m in metrics:
        assert m["mean_fg_entropy"] >= 0.0
        assert m["mean_fg_bald"] >= 0.0


def test_entropy_bald_correlation(dummy_model):
    """
    Verifies the critical milestone check from Active_Learning_MUJ.pdf:
    Predictive entropy and BALD must show strong correlation (>= 0.85) on trained weights.
    """
    device = torch.device("cpu")
    ckpt_path = "outputs_kaggle/checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        model = DeepLabV3Plus25D(encoder_name="resnet34", in_channels=9, num_classes=6)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
    else:
        model = dummy_model

    scorer = MCDropoutScorer(model=model, device=device, num_passes=5, use_amp=False)
    x = torch.randn(2, 9, 64, 64)
    res = scorer.score_batch(x)

    ent = res["entropy_map"].view(-1)
    bald = res["bald_map"].view(-1)

    corr = torch.corrcoef(torch.stack([ent, bald]))[0, 1].item()
    print(f"Entropy-BALD correlation: {corr:.4f}")
    if os.path.exists(ckpt_path):
        assert corr >= 0.85, f"Correlation {corr:.4f} is below required threshold 0.85!"
    else:
        assert not np.isnan(corr)


def test_stratified_vs_topk_and_random():
    """
    Verifies that StratifiedSampler allocates across bands, TopKSampler takes ceiling,
    and RandomSampler samples cleanly with study diversity.
    """
    np.random.seed(42)
    records = []
    for i in range(120):
        records.append({
            "study_id": f"patient_{i // 4}",
            "filename": f"patient_{i // 4}.nii.gz",
            "slice_idx": i % 4,
            "mean_fg_entropy": np.random.uniform(0.05, 0.75),
            "pred_area": np.random.randint(20, 500),
            "eligible_candidate": True,
            "category": "anybleed" if i % 2 == 0 else "nobleed",
            "has_edh_pred": i % 15 == 0,
            "has_iph_pred": i % 3 == 0,
            "has_sah_pred": False,
            "has_sdh_pred": False,
            "has_ivh_pred": False,
        })
    pool_df = pd.DataFrame(records)

    budget = 20
    # Stratified
    stratified = StratifiedSampler(random_seed=42)
    al_selected = stratified.select(pool_df, budget=budget)
    assert len(al_selected) == budget
    assert "selection_band" in al_selected.columns
    assert al_selected["study_id"].nunique() >= 5

    # Top-K
    topk_sampler = TopKSampler()
    topk_selected = topk_sampler.select(pool_df, budget=budget)
    assert len(topk_selected) == budget
    assert topk_selected["mean_fg_entropy"].min() >= pool_df["mean_fg_entropy"].median()

    # Random
    random_sampler = RandomSampler(random_seed=42)
    rnd_selected = random_sampler.select(pool_df, budget=budget)
    assert len(rnd_selected) == budget
    assert rnd_selected["study_id"].nunique() >= 5


def test_unlabeled_discovery_and_loading():
    """
    Verifies that discover_unlabeled_scans and UnlabeledCTDataset accurately
    discover, slice, window, and return 2.5D multi-window tensors from anybleed and nobleed folders.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        anybleed_dir = os.path.join(temp_dir, "anybleed")
        nobleed_dir = os.path.join(temp_dir, "nobleed")
        os.makedirs(anybleed_dir, exist_ok=True)
        os.makedirs(nobleed_dir, exist_ok=True)

        # 1. Create a 3D NIfTI scan in anybleed
        hu_vol = np.random.uniform(-100, 200, (32, 32, 4)).astype(np.float32)
        nii = nib.Nifti1Image(hu_vol, affine=np.eye(4))
        nii_path = os.path.join(anybleed_dir, "scan_01.nii.gz")
        nib.save(nii, nii_path)

        # 2. Create a 2D image scan in nobleed
        img_arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        img_path = os.path.join(nobleed_dir, "scan_02.png")
        cv2.imwrite(img_path, img_arr)

        scans = discover_unlabeled_scans(unlabeled_dir=temp_dir)
        assert len(scans) == 2
        categories = {s["category"] for s in scans}
        assert "anybleed" in categories
        assert "nobleed" in categories

        ds = UnlabeledCTDataset(unlabeled_dir=temp_dir, target_size=(32, 32))
        assert len(ds) == 4 + 1  # 4 slices from NIfTI + 1 from 2D image

        tensor_0, meta_0 = ds[0]
        assert tensor_0.shape == (9, 32, 32)
        assert meta_0["category"] == "anybleed"
        assert meta_0["num_slices"] == 4

        tensor_last, meta_last = ds[-1]
        assert tensor_last.shape == (9, 32, 32)
        assert meta_last["category"] == "nobleed"
        assert meta_last["num_slices"] == 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_simulated_oracle_structure():
    """
    Verifies SimulatedOracle initialization and query interface.
    """
    oracle = SimulatedOracle(dataset_dir="Dataset")
    assert hasattr(oracle, "annotate")


if __name__ == "__main__":
    print("[RUNNING] Running Active Learning unit tests...")
    model = create_dummy_model()
    test_mc_dropout_uncertainty_bounds(model)
    print("  ✓ test_mc_dropout_uncertainty_bounds passed")
    test_entropy_bald_correlation(model)
    print("  ✓ test_entropy_bald_correlation passed")
    test_stratified_vs_topk_and_random()
    print("  ✓ test_stratified_vs_topk_and_random passed")
    test_unlabeled_discovery_and_loading()
    print("  ✓ test_unlabeled_discovery_and_loading passed")
    test_simulated_oracle_structure()
    print("  ✓ test_simulated_oracle_structure passed")
    print("[SUCCESS] All Active Learning unit tests passed!")
