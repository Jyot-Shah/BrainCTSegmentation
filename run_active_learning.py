"""
End-to-End Active Learning Pipeline Runner for Brain CT Hemorrhage Segmentation.

Directly aligned with Active_Learning_MUJ.pdf:
1. Monte Carlo Dropout epistemic uncertainty scoring on unlabeled pool (anybleed / nobleed scans).
2. Fast HU pre-filtering (skips slices with max brain HU < 60 before GPU scoring).
3. Quality floors (min_entropy >= 0.10, min_pred_area >= 50 px) and patient study diversity capping.
4. Stratified uncertainty selection across 3 bands (Bands A, B, C) vs. Random Selection control.
5. Simulated oracle annotation and buffer-preserved fine-tuning (anti-catastrophic forgetting).
6. Generation of headline Two-Curve Label-Efficiency plots and inspection figures.
"""

import os
import sys
import argparse
import time
import torch
import pandas as pd
import numpy as np

from src.models.deeplab import DeepLabV3Plus25D
from src.active_learning.uncertainty import MCDropoutScorer
from src.active_learning.pool_scorer import PoolScorer
from src.active_learning.selector import StratifiedSampler, RandomSampler, TopKSampler, SimulatedOracle
from src.active_learning.loop import ActiveLearningManager
from src.active_learning.visualize import (
    plot_label_efficiency_curves,
    plot_entropy_drift,
    plot_uncertainty_overlay,
    plot_stratified_vs_topk,
)
from src.data.dataset import BrainCT25DDataset
from src.data.unlabeled import UnlabeledCTDataset, discover_unlabeled_scans
from src.data.transforms import CTTransform


def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning Pipeline for Intracranial Hemorrhage Segmentation")
    parser.add_argument("--checkpoint", type=str, default="outputs_kaggle/checkpoints/best_model.pth", help="Phase 2 baseline checkpoint")
    parser.add_argument("--unlabeled_dir", type=str, default=None, help="Directory containing 'anybleed' and 'nobleed' folders")
    parser.add_argument("--dataset_dir", type=str, default="Dataset", help="Main Dataset folder")
    parser.add_argument("--dev_manifest", type=str, default="data_splits/dev_manifest.csv", help="Fallback candidate pool manifest")
    parser.add_argument("--val_manifest", type=str, default="data_splits/val_manifest.csv", help="Validation cohort manifest")
    parser.add_argument("--train_manifest", type=str, default="data_splits/train_manifest.csv", help="Training cohort manifest")
    parser.add_argument("--train_slices", type=str, default="data_splits/train_slices.csv", help="Training cohort slices")
    parser.add_argument("--output_dir", type=str, default="outputs/active_learning", help="Output directory")
    parser.add_argument("--rounds", type=int, default=2, help="Number of Active Learning rounds")
    parser.add_argument("--budget", type=int, default=50, help="Newly annotated slices queried per round")
    parser.add_argument("--num_passes", type=int, default=10, help="Number of MC Dropout forward passes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for scoring and fine-tuning")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Fine-tune epochs per round")
    parser.add_argument("--buffer_size", type=int, default=300, help="Anti-forgetting buffer set size")
    parser.add_argument("--max_scans", type=int, default=60, help="Max unlabeled scans to sample (balanced across anybleed/nobleed)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Quick smoke test mode with minimal budget")
    parser.add_argument("--force_rescore", action="store_true", help="Force re-scoring the candidate pool")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 75)
    print("  ACTIVE LEARNING PIPELINE &bull; BRAIN CT HEMORRHAGE SEGMENTATION")
    print("=" * 75)
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Base Checkpoint:      {args.checkpoint}")
    print(f"Dataset Root:         {args.dataset_dir}")
    print(f"Active Rounds:        {args.rounds}")
    print(f"Query Budget / Round: {args.budget} slices")
    print(f"MC Dropout Passes:    {args.num_passes}")
    print(f"Fast Dev Run:         {args.fast_dev_run}")
    print("=" * 75)

    if args.fast_dev_run:
        print("[FAST DEV RUN] Running in accelerated verification mode...")
        args.rounds = 1
        args.budget = 10
        args.fine_tune_epochs = 1
        args.buffer_size = 50
        args.num_passes = 3

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load Baseline Model
    print(f"\n[STEP 1] Loading pre-trained DeepLabV3+ model from: {args.checkpoint}")
    model = DeepLabV3Plus25D(encoder_name="resnet34", in_channels=9, num_classes=6)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device)
    print(f"[STEP 1] Model loaded successfully with {len(model.dropouts)} active dropout layers.")

    # 2. Setup Epistemic Scorer
    print(f"\n[STEP 2] Initializing Monte Carlo Dropout Scorer (T={args.num_passes} passes)...")
    scorer = MCDropoutScorer(
        model=model,
        device=device,
        num_passes=args.num_passes,
        use_amp=True,
    )

    # 3. Score Unlabeled Candidate Pool
    print(f"\n[STEP 3] Scoring Unlabeled Candidate Pool...")
    pool_scorer = PoolScorer(
        scorer=scorer,
        dataset_dir=args.dataset_dir,
        unlabeled_dir=args.unlabeled_dir,
        manifest_csv=args.dev_manifest,
        batch_size=args.batch_size,
        min_pred_area=50,
        min_entropy=0.10,
        max_slices_per_study=2,
        output_dir=args.output_dir,
        max_scans=args.max_scans,
    )
    pool_df = pool_scorer.score_pool(force_rescore=args.force_rescore)

    # 4. Stratified Selection vs Top-K Comparison (Week 10 Deliverable)
    print(f"\n[STEP 4] Comparing Stratified Selection vs. Naive Top-K...")
    stratified_sampler = StratifiedSampler(random_seed=42)
    topk_sampler = TopKSampler()

    stratified_sample = stratified_sampler.select(pool_df, budget=min(args.budget, len(pool_df)))
    topk_sample = topk_sampler.select(pool_df, budget=min(args.budget, len(pool_df)))

    topk_fig_path = os.path.join(figures_dir, "stratified_vs_topk.png")
    plot_stratified_vs_topk(stratified_sample, topk_sample, save_path=topk_fig_path)

    # 5. Qualitative Epistemic Uncertainty Overlay
    print(f"\n[STEP 5] Generating Qualitative Uncertainty Inspection Overlay...")
    top_candidate = pool_df.sort_values(by="mean_fg_entropy", ascending=False).iloc[0]
    top_fname = str(top_candidate["filename"])
    top_sidx = int(top_candidate["slice_idx"])
    top_cat = top_candidate.get("category", "")

    overlay_path = os.path.join(figures_dir, "uncertainty_overlay.png")
    overlay_done = False

    # Check pool_scorer dataset first
    if hasattr(pool_scorer, "dataset") and pool_scorer.dataset is not None:
        idx = pool_scorer.dataset.find_slice_index(top_fname, top_sidx) if hasattr(pool_scorer.dataset, "find_slice_index") else None
        if idx is not None:
            item = pool_scorer.dataset[idx]
            img = item[0] if isinstance(item, tuple) else item
            img_tensor = img.unsqueeze(0).to(device)
            with torch.no_grad():
                res = scorer.score_batch(img_tensor)
            ct_slice = img[1, :, :].cpu().numpy()
            pred_slice = res["pred_masks"][0].cpu().numpy()
            entropy_slice = res["entropy_map"][0].cpu().numpy()
            gt_slice = np.zeros_like(pred_slice)

            plot_uncertainty_overlay(
                ct_image=ct_slice,
                ground_truth=gt_slice,
                pred_mask=pred_slice,
                entropy_map=entropy_slice,
                slice_meta={"filename": top_fname, "slice_idx": top_sidx, "category": top_cat},
                save_path=overlay_path,
            )
            overlay_done = True

    if not overlay_done and os.path.exists(args.dev_manifest):
        dev_dataset = BrainCT25DDataset(
            manifest_csv=args.dev_manifest,
            dataset_dir=args.dataset_dir,
            transform=CTTransform(is_training=False),
            cache_capacity=4,
        )
        for img, mask, meta in dev_dataset:
            if meta["filename"] == top_fname and meta["slice_idx"] == top_sidx:
                img_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    res = scorer.score_batch(img_tensor)
                ct_slice = img[1, :, :].cpu().numpy()
                gt_slice = mask.cpu().numpy()
                pred_slice = res["pred_masks"][0].cpu().numpy()
                entropy_slice = res["entropy_map"][0].cpu().numpy()

                plot_uncertainty_overlay(
                    ct_image=ct_slice,
                    ground_truth=gt_slice,
                    pred_mask=pred_slice,
                    entropy_map=entropy_slice,
                    slice_meta={"filename": top_fname, "slice_idx": top_sidx, "category": top_cat},
                    save_path=overlay_path,
                )
                break

    # 6. Execute Multi-Round Active Learning Benchmark vs. Random Control
    print(f"\n[STEP 6] Executing Multi-Round Active Learning Benchmark vs Random Control...")
    manager = ActiveLearningManager(
        base_checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        train_manifest=args.train_manifest,
        train_slices=args.train_slices,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        device=device,
        buffer_size=args.buffer_size,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
    )

    history_df = manager.run_benchmark(
        pool_df=pool_df,
        rounds=args.rounds,
        budget_per_round=args.budget,
    )

    # 7. Generate Headline Two-Curve Plot and Entropy Drift (Phase 5 Deliverable)
    print(f"\n[STEP 7] Generating Publication-Grade Benchmark Figures...")
    curves_path = os.path.join(figures_dir, "label_efficiency_curves.png")
    plot_label_efficiency_curves(history_df, save_path=curves_path)

    drift_path = os.path.join(figures_dir, "entropy_drift.png")
    plot_entropy_drift(pool_df, pool_df[pool_df["eligible_candidate"]], save_path=drift_path)

    print("\n" + "=" * 75)
    print("  ACTIVE LEARNING PIPELINE EXECUTION COMPLETE!")
    print("=" * 75)
    print("Results Summary Table:")
    print(history_df[["round", "strategy", "total_annotated_slices", "macro_dice", "edh_dice", "iph_dice", "ivh_dice"]])
    print(f"\nGenerated Deliverables:")
    print(f"  - History CSV:           {os.path.join(args.output_dir, 'active_learning_history.csv')}")
    print(f"  - Pool Scores:           {os.path.join(args.output_dir, 'al_pool_scores.csv')}")
    print(f"  - Scores Parquet:        {os.path.join(args.output_dir, 'scores.parquet')}")
    print(f"  - Selected for Training: {os.path.join(args.output_dir, 'selected_for_training.csv')}")
    print(f"  - Label Efficiency Plot: {curves_path}")
    print(f"  - Stratified vs Top-K:   {topk_fig_path}")
    print(f"  - Uncertainty Overlay:   {os.path.join(figures_dir, 'uncertainty_overlay.png')}")
    print(f"  - Fine-Tuned model_v2:   {os.path.join(args.output_dir, 'model_v2.pth')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
