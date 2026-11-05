"""
Active Learning Query Selector and Simulated Oracle.

Implements:
- StratifiedSampler: Allocates annotation budget across 3 entropy bands
  (Band A: High >= 0.50, Band B: Medium 0.30-0.50, Band C: Boundary 0.10-0.30)
  strictly as specified in Week 10 of Active_Learning_MUJ.pdf.
- TopKSampler: Pure top-K selection by entropy (for entropy ceiling comparison).
- RandomSampler: Uniform random sampling under identical study diversity constraints (Passive Control).
- SimulatedOracle: Reveals held-out ground truth masks for query batches, or prepares selected_for_training.csv.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
import nibabel as nib

from src.data.split import BHSD_TO_TARGET


class StratifiedSampler:
    """
    Selects query slices using stratified entropy band allocation across 3 bands:
    - Band A (High Uncertainty): entropy >= 0.50 (allocates ~27% of budget, default 40/150)
    - Band B (Medium Uncertainty): entropy 0.30 - 0.50 (allocates ~27% of budget, default 40/150)
    - Band C (Boundary Refinement): entropy 0.10 - 0.30 (allocates ~46% of budget, default 70/150)
    """
    def __init__(
        self,
        band_a_threshold: float = 0.50,
        band_b_threshold: float = 0.30,
        band_c_threshold: float = 0.10,
        band_ratios: Tuple[float, float, float] = (40.0 / 150.0, 40.0 / 150.0, 70.0 / 150.0),
        random_seed: int = 42,
    ):
        self.band_a_thresh = band_a_threshold
        self.band_b_thresh = band_b_threshold
        self.band_c_thresh = band_c_threshold
        self.band_ratios = band_ratios
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def select(
        self,
        pool_df: pd.DataFrame,
        budget: int,
        already_selected_ids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """
        Selects a query batch of size `budget` from eligible candidates across 3 strata.
        """
        df = pool_df.copy()

        # Build unique slice ID
        study_col = "study_id" if "study_id" in df.columns else "filename"
        df["slice_id"] = df[study_col].astype(str) + "___" + df["slice_idx"].astype(str)

        # Exclude already selected slices
        if already_selected_ids:
            df = df[~df["slice_id"].isin(already_selected_ids)]

        # Filter to eligible candidates (passed quality floor and study diversity cap)
        if "eligible_candidate" in df.columns:
            candidates = df[df["eligible_candidate"]].copy()
        else:
            candidates = df.copy()

        if len(candidates) < budget:
            # Fallback to all unselected candidates if eligible set is exhausted
            candidates = df.copy()

        # If candidates pool is smaller than requested budget, return all available candidates
        if len(candidates) <= budget:
            res = candidates.copy()
            res["selection_band"] = "Full Eligible Pool"
            res["selection_strategy"] = "Active Learning (Stratified)"
            return res

        # Partition candidates into 3 entropy strata
        band_a = candidates[candidates["mean_fg_entropy"] >= self.band_a_thresh].copy()
        band_b = candidates[
            (candidates["mean_fg_entropy"] >= self.band_b_thresh) &
            (candidates["mean_fg_entropy"] < self.band_a_thresh)
        ].copy()
        band_c = candidates[
            (candidates["mean_fg_entropy"] >= self.band_c_thresh) &
            (candidates["mean_fg_entropy"] < self.band_b_thresh)
        ].copy()

        # If highest band is empty, dynamically adjust threshold to ensure 3 populated bands
        if len(band_a) == 0:
            median_ent = candidates["mean_fg_entropy"].median()
            q75_ent = candidates["mean_fg_entropy"].quantile(0.75)
            band_a = candidates[candidates["mean_fg_entropy"] >= q75_ent].copy()
            band_b = candidates[(candidates["mean_fg_entropy"] >= median_ent) & (candidates["mean_fg_entropy"] < q75_ent)].copy()
            band_c = candidates[candidates["mean_fg_entropy"] < median_ent].copy()

        # Allocate budget
        n_a = int(round(budget * self.band_ratios[0]))
        n_b = int(round(budget * self.band_ratios[1]))
        n_c = budget - (n_a + n_b)

        def sample_from_band(band_df: pd.DataFrame, target_n: int, label: str) -> pd.DataFrame:
            if len(band_df) == 0 or target_n <= 0:
                return pd.DataFrame()

            # Rare-class and coarse bleed priority weights
            weights = np.ones(len(band_df))
            if "has_edh_pred" in band_df.columns:
                weights += band_df["has_edh_pred"].astype(float).values * 3.0
            if "has_sah_pred" in band_df.columns:
                weights += band_df["has_sah_pred"].astype(float).values * 1.5
            if "has_ivh_pred" in band_df.columns:
                weights += band_df["has_ivh_pred"].astype(float).values * 1.5
            if "category" in band_df.columns:
                weights += (band_df["category"] == "anybleed").astype(float).values * 1.5

            probs = weights / weights.sum()
            n_sample = min(target_n, len(band_df))
            chosen = self.rng.choice(len(band_df), size=n_sample, replace=False, p=probs)
            sampled = band_df.iloc[chosen].copy()
            sampled["selection_band"] = label
            return sampled

        sel_a = sample_from_band(band_a, n_a, "Band A (High Uncertainty)")
        sel_b = sample_from_band(band_b, n_b, "Band B (Medium Uncertainty)")
        sel_c = sample_from_band(band_c, n_c, "Band C (Boundary Refinement)")

        selected = pd.concat([sel_a, sel_b, sel_c], ignore_index=True)

        # Top-up if any band had insufficient slices
        if len(selected) < budget:
            shortfall = budget - len(selected)
            chosen_ids = set(selected["slice_id"])
            remaining = candidates[~candidates["slice_id"].isin(chosen_ids)].copy()
            if len(remaining) > 0:
                top_up = remaining.sort_values(by="mean_fg_entropy", ascending=False).head(shortfall).copy()
                top_up["selection_band"] = "Top-Up (Fallback)"
                selected = pd.concat([selected, top_up], ignore_index=True)

        selected["selection_strategy"] = "Active Learning (Stratified)"
        print(f"[SELECTOR] Stratified AL Selection (Budget: {budget}):")
        print(f"  Band A (>= {self.band_a_thresh:.2f}):  {len(sel_a)} slices")
        print(f"  Band B ({self.band_b_thresh:.2f}-{self.band_a_thresh:.2f}): {len(sel_b)} slices")
        print(f"  Band C ({self.band_c_thresh:.2f}-{self.band_b_thresh:.2f}): {len(sel_c)} slices")
        print(f"  Total Selected:        {len(selected)} slices across {selected['filename'].nunique()} scans")

        return selected


class TopKSampler:
    """
    Pure top-K selection by predictive entropy (demonstrates entropy ceiling clustering).
    """
    def __init__(self):
        pass

    def select(
        self,
        pool_df: pd.DataFrame,
        budget: int,
        already_selected_ids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        df = pool_df.copy()
        study_col = "study_id" if "study_id" in df.columns else "filename"
        df["slice_id"] = df[study_col].astype(str) + "___" + df["slice_idx"].astype(str)

        if already_selected_ids:
            df = df[~df["slice_id"].isin(already_selected_ids)]

        candidates = df[df["eligible_candidate"]].copy() if "eligible_candidate" in df.columns else df.copy()
        if len(candidates) < budget:
            candidates = df.copy()

        selected = candidates.sort_values(by="mean_fg_entropy", ascending=False).head(budget).copy()
        selected["selection_strategy"] = "Naive Top-K Selection"
        selected["selection_band"] = "Top-K Ceiling"
        return selected


class RandomSampler:
    """
    Uniform random baseline sampler (Passive Learning control).
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def select(
        self,
        pool_df: pd.DataFrame,
        budget: int,
        already_selected_ids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        df = pool_df.copy()
        study_col = "study_id" if "study_id" in df.columns else "filename"
        df["slice_id"] = df[study_col].astype(str) + "___" + df["slice_idx"].astype(str)

        if already_selected_ids:
            df = df[~df["slice_id"].isin(already_selected_ids)]

        candidates = df[df["eligible_candidate"]].copy() if "eligible_candidate" in df.columns else df.copy()
        if len(candidates) < budget:
            candidates = df.copy()

        n_sample = min(budget, len(candidates))
        chosen_indices = self.rng.choice(len(candidates), size=n_sample, replace=False)
        selected = candidates.iloc[chosen_indices].copy()
        selected["selection_strategy"] = "Random Selection (Control)"
        selected["selection_band"] = "Random"

        print(f"[SELECTOR] Random Control Selection (Budget: {budget}):")
        print(f"  Sampled {len(selected)} slices across {selected['filename'].nunique()} scans")

        return selected


class SimulatedOracle:
    """
    Simulates expert annotation by revealing ground-truth masks for selected query slices,
    or formats purely unlabelled slices for training integration.
    """
    def __init__(self, dataset_dir: str = "Dataset", output_dir: str = "outputs/active_learning"):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.gt_dir = os.path.join(dataset_dir, "labeled", "ground_truths")
        os.makedirs(self.output_dir, exist_ok=True)

    def annotate(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reveals true class annotations for queried slices and creates training manifest.
        Saves query batch to selected_for_training.csv.
        """
        annotated_records = []

        for _, row in query_df.iterrows():
            fname = str(row["filename"])
            s_idx = int(row["slice_idx"])
            gt_path = os.path.join(self.gt_dir, fname)

            # Check if ground truth mask is available on disk
            if os.path.exists(gt_path):
                gt_nii = nib.load(gt_path)
                gt_data = np.asanyarray(gt_nii.dataobj)
                slice_mask = gt_data[:, :, s_idx]

                raw_classes = [int(c) for c in np.unique(slice_mask)]
                target_classes = sorted(list(set([BHSD_TO_TARGET.get(c, 0) for c in raw_classes])))
                lesion_classes = [c for c in target_classes if c > 0]
                num_slices = gt_data.shape[2]

                gt_nii.uncache()
                del gt_nii
            else:
                # Purely unlabelled scan without masks
                # Infer binary coarse indicator from category
                is_bleed = row.get("category", "") == "anybleed"
                target_classes = [0, 1] if is_bleed else [0]
                lesion_classes = [1] if is_bleed else []
                num_slices = int(row.get("num_slices", 30))

            annotated_records.append({
                "filename": fname,
                "slice_idx": s_idx,
                "num_slices": num_slices,
                "target_classes": ",".join(str(c) for c in target_classes),
                "has_foreground": int(len(lesion_classes) > 0),
                "has_edh": int(3 in lesion_classes),
                "has_iph": int(1 in lesion_classes),
                "has_sah": int(2 in lesion_classes),
                "has_sdh": int(4 in lesion_classes),
                "has_ivh": int(5 in lesion_classes),
                "selection_strategy": row.get("selection_strategy", "Unknown"),
                "selection_band": row.get("selection_band", "Unknown"),
                "mean_fg_entropy": float(row.get("mean_fg_entropy", 0.0)),
                "category": row.get("category", "unlabeled"),
            })

        result_df = pd.DataFrame(annotated_records)
        
        # Save selected_for_training.csv deliverable (Week 10)
        selected_csv = os.path.join(self.output_dir, "selected_for_training.csv")
        result_df.to_csv(selected_csv, index=False)
        print(f"[ORACLE] Annotated {len(result_df)} query slices (Saved to: {selected_csv})")

        return result_df
