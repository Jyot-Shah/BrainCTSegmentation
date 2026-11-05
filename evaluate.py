"""
Evaluation Entrypoint Script for 2.5D Brain CT Hemorrhage Segmentation.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --config configs/train_config.yaml
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger
from src.data.transforms import CTTransform
from src.data.dataset import BrainCT25DDataset
from src.models.deeplab import DeepLabV3Plus25D
from src.evaluation.evaluate import evaluate_model_on_dataset, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2.5D DeepLabV3+ Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "dev"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger("evaluate_pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    # Manifest selection
    if args.split == "val":
        manifest_csv = cfg["data"]["val_manifest"]
    else:
        manifest_csv = cfg["data"]["dev_manifest"]

    dataset_dir = cfg["data"].get("dataset_dir", "Dataset")
    transform = CTTransform(is_training=False)

    dataset = BrainCT25DDataset(
        manifest_csv=manifest_csv,
        dataset_dir=dataset_dir,
        transform=transform,
        filter_empty_slices_prob=0.0,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    logger.info(f"Loaded {len(dataset)} slices from {args.split} split.")

    # Model
    model_cfg = cfg.get("model", {})
    model = DeepLabV3Plus25D(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        encoder_output_stride=model_cfg.get("encoder_output_stride", 8),
        decoder_atrous_rates=tuple(model_cfg.get("decoder_atrous_rates", (6, 12, 18))),
        in_channels=model_cfg.get("in_channels", 9),
        num_classes=model_cfg.get("num_classes", 6),
        decoder_dropout=float(model_cfg.get("decoder_dropout", 0.2)),
    )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    logger.info(f"Loading checkpoint weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_model_on_dataset(
        model=model,
        val_loader=data_loader,
        device=device,
        output_dir=cfg["output"].get("metrics_dir", "outputs/metrics"),
        figure_dir=cfg["output"].get("figures_dir", "outputs/figures"),
    )
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
