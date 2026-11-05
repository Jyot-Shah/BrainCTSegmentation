"""
Active Learning Pool Scorer and Quality Filtering Engine.

Implements:
- Scoring of unlabeled candidate pool (anybleed and nobleed scans).
- Fallback support for development cohort manifest if unlabeled folder is pending.
- Fast HU pre-filtering (skips slices with max brain-window HU < 60 before GPU scoring).
- Batch-wise MC Dropout uncertainty extraction (T passes).
- Quality floors (min_pred_area >= 50 px, min_entropy >= 0.10).
- Study-level diversity ranking (capping adjacent slices per patient study).
- Caching to both scores.parquet and scores.csv for full pipeline compatibility.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.dataset import BrainCT25DDataset
from src.data.transforms import CTTransform
from src.data.unlabeled import UnlabeledCTDataset, discover_unlabeled_scans
from src.active_learning.uncertainty import MCDropoutScorer


class PoolScorer:
    """
    Scores and filters an unlabeled candidate pool using epistemic uncertainty.
    """
    def __init__(
        self,
        scorer: MCDropoutScorer,
        dataset_dir: str = "Dataset",
        unlabeled_dir: Optional[str] = None,
        manifest_csv: Optional[str] = "data_splits/dev_manifest.csv",
        batch_size: int = 4,
        num_workers: int = 0,
        min_pred_area: int = 50,
        min_entropy: float = 0.10,
        max_slices_per_study: int = 2,
        output_dir: str = "outputs/active_learning",
        max_scans: Optional[int] = None,
    ):
        self.scorer = scorer
        self.dataset_dir = dataset_dir
        self.unlabeled_dir = unlabeled_dir
        self.manifest_csv = manifest_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_pred_area = min_pred_area
        self.min_entropy = min_entropy
        self.max_slices_per_study = max_slices_per_study
        self.output_dir = output_dir
        self.max_scans = max_scans
        self.dataset: Optional[Dataset] = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.scores_csv_path = os.path.join(self.output_dir, "al_pool_scores.csv")
        self.scores_parquet_path = os.path.join(self.output_dir, "scores.parquet")

    def _resolve_dataset(self) -> Tuple[Dataset, str]:
        """
        Discovers unlabeled dataset from 'anybleed' and 'nobleed' folders.
        Falls back to manifest_csv if no scans are found.
        """
        scans = discover_unlabeled_scans(
            unlabeled_dir=self.unlabeled_dir,
            dataset_dir=self.dataset_dir,
            max_scans=self.max_scans,
        )

        if len(scans) > 0:
            anybleed_cnt = sum(1 for s in scans if s["category"] == "anybleed")
            nobleed_cnt = sum(1 for s in scans if s["category"] == "nobleed")
            print(f"[DATASET] Discovered unlabeled scans: {len(scans)} total ({anybleed_cnt} anybleed, {nobleed_cnt} nobleed).")
            ds = UnlabeledCTDataset(
                unlabeled_dir=self.unlabeled_dir,
                dataset_dir=self.dataset_dir,
                cache_capacity=8,
                max_scans=self.max_scans,
            )
            self.dataset = ds
            return ds, "unlabeled_folders"

        # Fallback to dev manifest if available
        if self.manifest_csv and os.path.exists(self.manifest_csv):
            print(f"[DATASET] Unlabeled scans not yet present in anybleed/nobleed. Using dev cohort from: {self.manifest_csv}")
            ds = BrainCT25DDataset(
                manifest_csv=self.manifest_csv,
                dataset_dir=self.dataset_dir,
                transform=CTTransform(is_training=False),
                cache_capacity=8,
                filter_empty_slices_prob=0.0,
            )
            self.dataset = ds
            return ds, "manifest_fallback"

        raise FileNotFoundError(
            f"No scans found in '{self.dataset_dir}/unlabeled/anybleed' or 'nobleed', and manifest '{self.manifest_csv}' does not exist. "
            f"Please place your unlabelled scans in Dataset/unlabeled/anybleed and Dataset/unlabeled/nobleed."
        )

    def score_pool(self, force_rescore: bool = False) -> pd.DataFrame:
        """
        Scores all slices in the candidate pool.
        Returns:
            pd.DataFrame with uncertainty metrics, quality flags, and diversity eligibility.
        """
        # Resolve dataset first so we know mode and can reference self.dataset
        dataset, dataset_mode = self._resolve_dataset()

        # Cache validation check
        if not force_rescore and (os.path.exists(self.scores_parquet_path) or os.path.exists(self.scores_csv_path)):
            load_path = self.scores_parquet_path if os.path.exists(self.scores_parquet_path) else self.scores_csv_path
            try:
                if load_path.endswith(".parquet"):
                    cached_df = pd.read_parquet(load_path)
                else:
                    cached_df = pd.read_csv(load_path)

                # Verify cache compatibility with current pool mode
                cache_is_unlabeled = "category" in cached_df.columns and set(cached_df["category"].dropna().unique()).intersection({"anybleed", "nobleed"})
                if dataset_mode == "unlabeled_folders" and not cache_is_unlabeled:
                    print(f"[CACHE] Outdated fallback cache detected; re-scoring new unlabelled anybleed/nobleed scans...")
                else:
                    print(f"[CACHE] Loading cached pool uncertainty scores ({len(cached_df)} slices) from: {load_path}")
                    return self._apply_quality_and_diversity_filters(cached_df)
            except Exception as e:
                print(f"[CACHE] Failed to load cache: {e}. Scoring anew...")

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        all_records = []
        print(f"[SCORING] Running MC Dropout scoring on {len(dataset)} slices (Mode: {dataset_mode})...")

        with tqdm(total=len(dataset), desc="Scoring AL Pool") as pbar:
            for batch_data in loader:
                if dataset_mode == "unlabeled_folders":
                    images, metas = batch_data
                    masks = None
                else:
                    images, masks, metas = batch_data

                b = images.shape[0]

                # Check HU pre-filter: skip slices where max brain HU < 60 BEFORE GPU scoring
                if "passes_hu_prefilter" in metas:
                    prefilter_flags = metas["passes_hu_prefilter"]
                    if isinstance(prefilter_flags, torch.Tensor):
                        prefilter_flags = prefilter_flags.cpu().numpy()
                else:
                    center_brain = images[:, 1, :, :]
                    max_vals = center_brain.view(b, -1).max(dim=1)[0].numpy()
                    prefilter_flags = (max_vals >= 0.70)  # ~60 HU in brain window

                # Fast HU pre-filter: ONLY score slices that passed HU pre-filter on GPU!
                passed_indices = [j for j, flag in enumerate(prefilter_flags) if bool(flag)]
                passed_metrics: Dict[int, Dict[str, Any]] = {}

                if len(passed_indices) > 0:
                    scored_images = images[passed_indices]
                    with torch.no_grad():
                        res = self.scorer.score_batch(scored_images)
                    sub_metrics = res["slice_metrics"]
                    for idx_in_sub, original_idx in enumerate(passed_indices):
                        passed_metrics[original_idx] = sub_metrics[idx_in_sub]

                for i in range(b):
                    fname = metas["filename"][i] if isinstance(metas["filename"], list) else metas["filename"][i]
                    s_idx = int(metas["slice_idx"][i]) if isinstance(metas["slice_idx"], torch.Tensor) else metas["slice_idx"][i]
                    study_id = metas.get("study_id", [fname]*b)[i] if "study_id" in metas else os.path.splitext(fname)[0]
                    category = metas.get("category", ["unlabeled"]*b)[i] if "category" in metas else "unlabeled"

                    passed_hu = (i in passed_metrics)
                    if passed_hu:
                        m = passed_metrics[i]
                    else:
                        m = {
                            "mean_fg_entropy": 0.0,
                            "mean_fg_bald": 0.0,
                            "mean_fg_var": 0.0,
                            "mean_fg_vr": 0.0,
                            "max_entropy": 0.0,
                            "max_bald": 0.0,
                            "mean_whole_entropy": 0.0,
                            "pred_area": 0,
                            "detected_classes": [],
                            "has_edh_pred": False,
                            "has_iph_pred": False,
                            "has_sah_pred": False,
                            "has_sdh_pred": False,
                            "has_ivh_pred": False,
                        }

                    record = {
                        "filename": str(fname),
                        "study_id": str(study_id),
                        "slice_idx": int(s_idx),
                        "category": str(category),
                        "passes_hu_prefilter": bool(passed_hu),
                        "mean_fg_entropy": m["mean_fg_entropy"],
                        "mean_fg_bald": m["mean_fg_bald"],
                        "mean_fg_var": m["mean_fg_var"],
                        "mean_fg_vr": m["mean_fg_vr"],
                        "max_entropy": m["max_entropy"],
                        "max_bald": m["max_bald"],
                        "mean_whole_entropy": m["mean_whole_entropy"],
                        "pred_area": m["pred_area"],
                        "detected_classes": str(m["detected_classes"]),
                        "has_edh_pred": bool(m["has_edh_pred"]),
                        "has_iph_pred": bool(m["has_iph_pred"]),
                        "has_sah_pred": bool(m["has_sah_pred"]),
                        "has_sdh_pred": bool(m["has_sdh_pred"]),
                        "has_ivh_pred": bool(m["has_ivh_pred"]),
                    }

                    # Ground truth flags if available (for simulated oracle evaluation)
                    if masks is not None:
                        true_classes = torch.unique(masks[i]).cpu().numpy().tolist()
                        true_lesions = [c for c in true_classes if c > 0]
                        record["gt_has_foreground"] = len(true_lesions) > 0
                        record["gt_has_edh"] = 3 in true_lesions
                        record["gt_has_iph"] = 1 in true_lesions
                        record["gt_has_sah"] = 2 in true_lesions
                        record["gt_has_sdh"] = 4 in true_lesions
                        record["gt_has_ivh"] = 5 in true_lesions
                        record["gt_classes"] = str(true_lesions)

                    all_records.append(record)

                pbar.update(b)

        df = pd.DataFrame(all_records)

        # Save to both formats
        df.to_csv(self.scores_csv_path, index=False)
        try:
            df.to_parquet(self.scores_parquet_path, index=False)
            print(f"[CACHE] Saved pool scores ({len(df)} slices) to {self.scores_parquet_path} and {self.scores_csv_path}")
        except Exception as e:
            print(f"[CACHE] Saved pool scores to {self.scores_csv_path} (parquet notice: {e})")

        return self._apply_quality_and_diversity_filters(df)

    def _apply_quality_and_diversity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies clinical quality floors and study diversity ranking:
        - Quality Floor: min_pred_area >= 50 px AND min_entropy >= 0.10
        - Study Diversity: Rank slices within study by entropy.
        - Eligible Candidates: Slices meeting quality floor. Diversity cap is applied during selection.
        """
        df = df.copy()

        # 1. Quality Floor
        df["passes_quality_floor"] = (
            (df["pred_area"] >= self.min_pred_area) &
            (df["mean_fg_entropy"] >= self.min_entropy)
        )

        # 2. Within-study ranking
        group_col = "study_id" if "study_id" in df.columns else "filename"
        df["study_entropy_rank"] = df.groupby(group_col)["mean_fg_entropy"].rank(
            ascending=False, method="first"
        ).astype(int)

        # 3. Diversity cap indicator
        df["passes_diversity_cap"] = df["study_entropy_rank"] <= self.max_slices_per_study

        # 4. Overall candidate eligibility (meets clinical quality floor)
        df["eligible_candidate"] = df["passes_quality_floor"]

        total_slices = len(df)
        quality_eligible = int(df["passes_quality_floor"].sum())
        cap_rank_slices = int((df["passes_quality_floor"] & df["passes_diversity_cap"]).sum())

        print(f"[FILTERS] Active Learning Pool Summary:")
        print(f"  Total Slices Scored:       {total_slices}")
        print(f"  Passed Quality Floor:      {quality_eligible} ({quality_eligible/max(1, total_slices)*100:.1f}%)")
        print(f"  Rank <= {self.max_slices_per_study} in Study:        {cap_rank_slices}")
        if "category" in df.columns:
            any_cnt = (df["category"] == "anybleed").sum()
            no_cnt = (df["category"] == "nobleed").sum()
            print(f"  Categories in Pool:        {any_cnt} anybleed, {no_cnt} nobleed")

        # Save eligible candidates
        eligible_path = os.path.join(self.output_dir, "eligible_candidates.csv")
        df[df["eligible_candidate"]].to_csv(eligible_path, index=False)

        return df
