"""
Active Learning Iterative Execution Loop and Buffer Retraining Manager.

Implements:
- Round-over-round active query acquisition.
- Buffer set assembly to prevent catastrophic forgetting (500-1000 slices stratified).
- Fine-tuning of both Active Learning and Random Control models.
- Held-out validation evaluation and tracking of per-class Dice metrics.
- Low VRAM footprint (AMP fp16, cache cleanup).
"""

import os
import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import nibabel as nib

from src.models.deeplab import DeepLabV3Plus25D
from src.training.losses import MulticlassDiceLoss
from src.data.dataset import BrainCT25DDataset
from src.data.transforms import (
    CTTransform,
    build_25d_input,
    normalize_channels,
)
from src.active_learning.selector import StratifiedSampler, RandomSampler, SimulatedOracle
from src.active_learning.pool_scorer import PoolScorer


class ActiveLearningSliceDataset(Dataset):
    """
    Fast slice dataset for fine-tuning on active learning query slices + anti-forgetting buffer set.
    Handles both labeled BHSD scans and newly queried unlabeled scans.
    """
    def __init__(
        self,
        slice_df: pd.DataFrame,
        dataset_dir: str = "Dataset",
        transform: Optional[CTTransform] = None,
    ):
        self.slice_df = slice_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform or CTTransform(is_training=True)

        self.img_dir = os.path.join(dataset_dir, "labeled", "images")
        self.gt_dir = os.path.join(dataset_dir, "labeled", "ground_truths")

        # Cache for loaded volumes
        self._vol_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.slice_df)

    def _load_volume(self, fname: str) -> Tuple[np.ndarray, np.ndarray]:
        if fname in self._vol_cache:
            return self._vol_cache[fname]

        img_path = os.path.join(self.img_dir, fname)
        gt_path = os.path.join(self.gt_dir, fname)

        # Handle unlabeled files if stored elsewhere
        if not os.path.exists(img_path):
            for candidate in [
                os.path.join(self.dataset_dir, "unlabeled", "anybleed", fname),
                os.path.join(self.dataset_dir, "unlabeled", "nobleed", fname),
                os.path.join(self.dataset_dir, "anybleed", fname),
                os.path.join(self.dataset_dir, "nobleed", fname),
            ]:
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if os.path.exists(img_path) and img_path.endswith((".nii", ".nii.gz")):
            img_nii = nib.load(img_path)
            img_data = np.asanyarray(img_nii.dataobj).astype(np.float32)
            img_nii.uncache()
            del img_nii
        else:
            img_data = np.zeros((512, 512, 30), dtype=np.float32)

        if os.path.exists(gt_path):
            gt_nii = nib.load(gt_path)
            gt_data = np.asanyarray(gt_nii.dataobj).astype(np.int64)
            gt_nii.uncache()
            del gt_nii
        else:
            # If no ground truth mask on disk, synthesize empty/background mask
            gt_data = np.zeros(img_data.shape, dtype=np.int64)

        if len(self._vol_cache) >= 12:
            self._vol_cache.pop(next(iter(self._vol_cache)))
        self._vol_cache[fname] = (img_data, gt_data)
        return img_data, gt_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.slice_df.iloc[idx]
        fname = str(row["filename"])
        z = int(row["slice_idx"])

        img_vol, gt_vol = self._load_volume(fname)
        total_z = img_vol.shape[2] if img_vol.ndim == 3 else 1

        z_prev = max(0, z - 1)
        z_curr = min(total_z - 1, max(0, z))
        z_next = min(total_z - 1, z + 1)

        s_prev = img_vol[:, :, z_prev] if img_vol.ndim == 3 else img_vol
        s_curr = img_vol[:, :, z_curr] if img_vol.ndim == 3 else img_vol
        s_next = img_vol[:, :, z_next] if img_vol.ndim == 3 else img_vol

        slices_hu = np.stack([s_prev, s_curr, s_next], axis=0).astype(np.float32)
        mask = gt_vol[:, :, z_curr].astype(np.int64) if gt_vol.ndim == 3 else np.zeros((512, 512), dtype=np.int64)

        norm_9ch, processed_mask = self.transform(slices_hu, mask)
        t_img = torch.from_numpy(norm_9ch).float()
        t_mask = torch.from_numpy(processed_mask).long()

        return t_img, t_mask, {"filename": fname, "slice_idx": z}


class ActiveLearningManager:
    """
    Executes and tracks multi-round Active Learning vs. Random Selection experiments.
    """
    def __init__(
        self,
        base_checkpoint_path: str,
        dataset_dir: str = "Dataset",
        train_manifest: str = "data_splits/train_manifest.csv",
        train_slices: str = "data_splits/train_slices.csv",
        val_manifest: str = "data_splits/val_manifest.csv",
        output_dir: str = "outputs/active_learning",
        device: Optional[torch.device] = None,
        buffer_size: int = 300,
        fine_tune_epochs: int = 3,
        fine_tune_lr: float = 1e-5,
        batch_size: int = 4,
        random_seed: int = 42,
    ):
        self.base_checkpoint_path = base_checkpoint_path
        self.dataset_dir = dataset_dir
        self.train_manifest = train_manifest
        self.train_slices = train_slices
        self.val_manifest = val_manifest
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr
        self.batch_size = batch_size
        self.random_seed = random_seed

        os.makedirs(self.output_dir, exist_ok=True)
        self.oracle = SimulatedOracle(dataset_dir=dataset_dir, output_dir=output_dir)
        self.rng = np.random.default_rng(random_seed)

        # Pre-select stratified buffer set from original training pool to prevent forgetting
        self.buffer_df = self._create_buffer_set()

        # Validation DataLoader
        val_ds = BrainCT25DDataset(
            manifest_csv=self.val_manifest,
            dataset_dir=self.dataset_dir,
            transform=CTTransform(is_training=False),
            cache_capacity=8,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _create_buffer_set(self) -> pd.DataFrame:
        """
        Samples a balanced buffer set from original training slices to prevent catastrophic forgetting.
        """
        df_train = pd.read_csv(self.train_slices)
        # Prioritize rare lesion slices (EDH, SAH, IVH) in the buffer set
        rare_slices = df_train[
            (df_train["has_edh"] == 1) | (df_train["has_sah"] == 1) | (df_train["has_ivh"] == 1)
        ].copy()

        n_rare = min(len(rare_slices), int(self.buffer_size * 0.6))
        n_common = max(0, self.buffer_size - n_rare)

        sampled_rare = rare_slices.sample(n=n_rare, random_state=self.random_seed) if n_rare > 0 else pd.DataFrame()
        remaining = df_train[~df_train.index.isin(sampled_rare.index)]
        sampled_common = remaining.sample(n=min(len(remaining), n_common), random_state=self.random_seed)

        buffer = pd.concat([sampled_rare, sampled_common], ignore_index=True)
        print(f"[BUFFER] Prepared anti-forgetting buffer set with {len(buffer)} slices ({len(sampled_rare)} rare lesion slices).")
        return buffer

    def load_base_model(self) -> DeepLabV3Plus25D:
        """
        Loads the Phase 2 baseline model weights.
        """
        model = DeepLabV3Plus25D(encoder_name="resnet34", in_channels=9, num_classes=6)
        ckpt = torch.load(self.base_checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        model.to(self.device)
        return model

    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluates model on the strictly held-out validation cohort (924 continuous slices).
        """
        model.eval()
        model.to(self.device)

        confusion = np.zeros((6, 6), dtype=np.int64)
        calibrated_thresholds = [0.5, 0.50, 0.35, 0.30, 0.35, 0.45]

        with torch.no_grad():
            for images, masks, _ in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = model(images)

                z_0 = logits[:, 0:1, :, :]
                rel_logits = logits - z_0
                lesion_probs = torch.sigmoid(rel_logits[:, 1:, :, :])
                bg_prob = torch.clamp(1.0 - lesion_probs.max(dim=1, keepdim=True)[0], min=0.0, max=1.0)
                probs = torch.cat([bg_prob, lesion_probs], dim=1)

                pred_scores = torch.zeros_like(probs)
                pred_scores[:, 0, :, :] = probs[:, 0, :, :]
                for c in range(1, 6):
                    pred_scores[:, c, :, :] = probs[:, c, :, :] / calibrated_thresholds[c]

                preds = pred_scores.argmax(dim=1)

                p_flat = preds.cpu().numpy().flatten()
                t_flat = masks.cpu().numpy().flatten()
                
                indices = t_flat * 6 + p_flat
                bincounts = np.bincount(indices, minlength=36)
                confusion += bincounts.reshape(6, 6)

        dice_scores = {}
        target_names = ["Background", "IPH", "SAH", "EDH", "SDH", "IVH"]
        macro_dice_sum = 0.0

        for c in range(6):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            denom = 2 * tp + fp + fn
            dice = (2.0 * tp / denom) if denom > 0 else 0.0
            dice_scores[target_names[c]] = float(dice)
            if c > 0:
                macro_dice_sum += dice

        macro_dice = macro_dice_sum / 5.0
        dice_scores["Macro_Dice"] = float(macro_dice)
        return dice_scores

    def fine_tune_model(
        self,
        model: nn.Module,
        train_df: pd.DataFrame,
        epochs: Optional[int] = None,
    ) -> nn.Module:
        """
        Fine-tunes model on active training set (new query slices + buffer set) at LR 1e-5.
        """
        epochs = epochs or self.fine_tune_epochs
        model.train()
        model.to(self.device)

        dataset = ActiveLearningSliceDataset(
            slice_df=train_df,
            dataset_dir=self.dataset_dir,
            transform=CTTransform(is_training=True),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.fine_tune_lr, weight_decay=1e-4)
        criterion = MulticlassDiceLoss(num_classes=6, smooth=1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        for _ in range(epochs):
            for images, masks, _ in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return model

    def run_benchmark(
        self,
        pool_df: pd.DataFrame,
        rounds: int = 2,
        budget_per_round: int = 50,
    ) -> pd.DataFrame:
        """
        Executes Active Learning (Stratified) vs. Random Selection (Control) benchmark.
        """
        history_records = []

        # ================= ROUND 0: BASELINE =================
        print("\n" + "="*70)
        print("  ACTIVE LEARNING BENCHMARK &bull; ROUND 0 (PRE-TRAINED BASELINE)")
        print("="*70)
        base_model = self.load_base_model()
        baseline_metrics = self.evaluate_model(base_model)

        print(f"[ROUND 0] Baseline Macro Dice: {baseline_metrics['Macro_Dice']:.4f}")
        for c in ["IPH", "SAH", "EDH", "SDH", "IVH"]:
            print(f"  {c}: {baseline_metrics[c]:.4f}", end=" | ")
        print()

        for strategy in ["Active Learning (Stratified)", "Random Selection (Control)"]:
            rec = {
                "round": 0,
                "strategy": strategy,
                "new_slices": 0,
                "total_annotated_slices": 0,
                "macro_dice": baseline_metrics["Macro_Dice"],
                "edh_dice": baseline_metrics["EDH"],
                "iph_dice": baseline_metrics["IPH"],
                "sah_dice": baseline_metrics["SAH"],
                "sdh_dice": baseline_metrics["SDH"],
                "ivh_dice": baseline_metrics["IVH"],
            }
            history_records.append(rec)

        # Active Learning and Random state tracking
        al_selected_ids = set()
        rnd_selected_ids = set()
        al_cum_annotated = []
        rnd_cum_annotated = []

        al_model = copy.deepcopy(base_model)
        rnd_model = copy.deepcopy(base_model)

        al_sampler = StratifiedSampler(random_seed=self.random_seed)
        rnd_sampler = RandomSampler(random_seed=self.random_seed)

        # ================= SUBSEQUENT ROUNDS =================
        for r in range(1, rounds + 1):
            print("\n" + "="*70)
            print(f"  ACTIVE LEARNING BENCHMARK &bull; ROUND {r} (BUDGET: {budget_per_round} SLICES)")
            print("="*70)

            # 1. Stratified Active Learning Selection
            print(f"\n[AL BRANCH] Selecting Round {r} slices via Stratified Uncertainty Sampler...")
            al_query = al_sampler.select(pool_df, budget=budget_per_round, already_selected_ids=al_selected_ids)
            study_col = "study_id" if "study_id" in al_query.columns else "filename"
            al_selected_ids.update(al_query[study_col].astype(str) + "___" + al_query["slice_idx"].astype(str))
            al_annotated = self.oracle.annotate(al_query)
            al_cum_annotated.append(al_annotated)
            al_train_df = pd.concat([self.buffer_df] + al_cum_annotated, ignore_index=True)

            # 2. Random Selection Branch (Control)
            print(f"\n[RANDOM BRANCH] Selecting Round {r} slices via Uniform Random Sampler...")
            rnd_query = rnd_sampler.select(pool_df, budget=budget_per_round, already_selected_ids=rnd_selected_ids)
            rnd_study_col = "study_id" if "study_id" in rnd_query.columns else "filename"
            rnd_selected_ids.update(rnd_query[rnd_study_col].astype(str) + "___" + rnd_query["slice_idx"].astype(str))
            rnd_annotated = self.oracle.annotate(rnd_query)
            rnd_cum_annotated.append(rnd_annotated)
            rnd_train_df = pd.concat([self.buffer_df] + rnd_cum_annotated, ignore_index=True)

            # 3. Retrain AL Model
            print(f"\n[RETRAIN AL] Fine-tuning Active Learning model on {len(al_train_df)} slices...")
            al_model = self.fine_tune_model(al_model, al_train_df)
            al_metrics = self.evaluate_model(al_model)

            # 4. Retrain Random Model
            print(f"\n[RETRAIN RND] Fine-tuning Random Control model on {len(rnd_train_df)} slices...")
            rnd_model = self.fine_tune_model(rnd_model, rnd_train_df)
            rnd_metrics = self.evaluate_model(rnd_model)

            total_slices_annotated = r * budget_per_round

            print(f"\n[ROUND {r} RESULTS] (Annotated Slices: {total_slices_annotated}):")
            print(f"  Active Learning Macro Dice: {al_metrics['Macro_Dice']:.4f} (EDH: {al_metrics['EDH']:.4f}, IPH: {al_metrics['IPH']:.4f})")
            print(f"  Random Control  Macro Dice: {rnd_metrics['Macro_Dice']:.4f} (EDH: {rnd_metrics['EDH']:.4f}, IPH: {rnd_metrics['IPH']:.4f})")
            delta = al_metrics['Macro_Dice'] - rnd_metrics['Macro_Dice']
            print(f"  --> AL Advantage Gap: {delta:+.4f} ({delta*100:+.2f}%)")

            history_records.append({
                "round": r,
                "strategy": "Active Learning (Stratified)",
                "new_slices": budget_per_round,
                "total_annotated_slices": total_slices_annotated,
                "macro_dice": al_metrics["Macro_Dice"],
                "edh_dice": al_metrics["EDH"],
                "iph_dice": al_metrics["IPH"],
                "sah_dice": al_metrics["SAH"],
                "sdh_dice": al_metrics["SDH"],
                "ivh_dice": al_metrics["IVH"],
            })

            history_records.append({
                "round": r,
                "strategy": "Random Selection (Control)",
                "new_slices": budget_per_round,
                "total_annotated_slices": total_slices_annotated,
                "macro_dice": rnd_metrics["Macro_Dice"],
                "edh_dice": rnd_metrics["EDH"],
                "iph_dice": rnd_metrics["IPH"],
                "sah_dice": rnd_metrics["SAH"],
                "sdh_dice": rnd_metrics["SDH"],
                "ivh_dice": rnd_metrics["IVH"],
            })

        history_df = pd.DataFrame(history_records)
        history_csv_path = os.path.join(self.output_dir, "active_learning_history.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"\n[SAVED] Benchmark history saved to: {history_csv_path}")

        # Save model_v2 checkpoint (Week 11 deliverable)
        model_v2_path = os.path.join(self.output_dir, "model_v2.pth")
        torch.save({"model_state_dict": al_model.state_dict(), "metrics": al_metrics}, model_v2_path)
        print(f"[SAVED] Fine-tuned model_v2 checkpoint saved to: {model_v2_path}")

        return history_df
