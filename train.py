"""
Main Training Script for 2.5D DeepLabV3+ Brain CT Hemorrhage Segmentation.

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --smoke-test
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

if os.name == "nt":
    try:
        import ctypes
        ctypes.cdll.msvcrt._setmaxstdio(2048)
    except Exception:
        pass

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.transforms import CTTransform
from src.data.dataset import BrainCT25DDataset
from src.data.sampler import create_class_aware_sampler
from src.models.deeplab import DeepLabV3Plus25D, verify_forward_and_gradient
from src.training.losses import CombinedDiceAndCrossEntropyLoss, compute_class_weights_from_manifest
from src.training.trainer import Trainer
from src.evaluation.evaluate import evaluate_model_on_dataset, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2.5D DeepLabV3+ for Brain CT Hemorrhage Segmentation")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config file")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test to verify gradient flow and pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger("train_pipeline")
    logger.info(f"Loaded configuration from: {args.config}")

    # Set reproducibility seeds
    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computing device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & Transforms
    aug_cfg = cfg.get("augmentation", {})
    train_transform = CTTransform(
        is_training=True,
        hflip_prob=aug_cfg.get("hflip_prob", 0.5),
        rotate_deg=aug_cfg.get("rotate_deg", 15.0),
        intensity_jitter_hu=aug_cfg.get("intensity_jitter_hu", 20.0),
    )
    val_transform = CTTransform(is_training=False)

    train_manifest = cfg["data"]["train_manifest"]
    val_manifest = cfg["data"]["val_manifest"]
    dataset_dir = cfg["data"].get("dataset_dir", "Dataset")

    filter_empty_prob = cfg["data"].get("filter_empty_slices_prob", 0.0)
    if args.smoke_test:
        filter_empty_prob = 0.8  # Further accelerate smoke test

    logger.info("Initializing datasets...")
    train_dataset = BrainCT25DDataset(
        manifest_csv=train_manifest,
        dataset_dir=dataset_dir,
        transform=train_transform,
        filter_empty_slices_prob=filter_empty_prob,
    )
    val_dataset = BrainCT25DDataset(
        manifest_csv=val_manifest,
        dataset_dir=dataset_dir,
        transform=val_transform,
        filter_empty_slices_prob=0.0,  # Never filter validation slices
    )

    logger.info(f"Indexed {len(train_dataset)} training slices and {len(val_dataset)} validation slices.")

    # 2. Sampler & DataLoader
    batch_size = args.batch_size or cfg["data"].get("batch_size", 4)
    num_workers = cfg["data"].get("num_workers", 0)

    sampler_cfg = cfg.get("minority_sampling", {})
    if sampler_cfg.get("enabled", True) and not args.smoke_test:
        logger.info("Applying minority-aware class sampler for SDH, SAH, and EDH...")
        train_sampler = create_class_aware_sampler(
            train_dataset.slice_indices,
            weight_edh=float(sampler_cfg.get("weight_edh", 12.0)),
            weight_sdh=float(sampler_cfg.get("weight_sdh", 6.0)),
            weight_sah=float(sampler_cfg.get("weight_sah", 5.0)),
            weight_iph=float(sampler_cfg.get("weight_iph", 2.0)),
            weight_ivh=float(sampler_cfg.get("weight_ivh", 2.5)),
            weight_bg=float(sampler_cfg.get("weight_bg", 0.5)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # 3. Model
    model_cfg = cfg.get("model", {})
    decoder_dropout = float(model_cfg.get("decoder_dropout", 0.2))
    logger.info(f"Building DeepLabV3+ with {model_cfg.get('encoder_name', 'resnet34')} encoder and dropout={decoder_dropout}...")

    model = DeepLabV3Plus25D(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        encoder_output_stride=model_cfg.get("encoder_output_stride", 8),
        decoder_atrous_rates=tuple(model_cfg.get("decoder_atrous_rates", (6, 12, 18))),
        in_channels=model_cfg.get("in_channels", 9),
        num_classes=model_cfg.get("num_classes", 6),
        decoder_dropout=decoder_dropout,
    )
    logger.info(f"Model validated successfully with {len(model.dropouts)} dropout layer(s).")

    # Smoke test forward and gradient check
    logger.info("Running forward pass and gradient flow verification...")
    verify_forward_and_gradient(model, device, in_channels=9, num_classes=6, input_size=(128, 128))

    # 4. Losses & Class Weights
    loss_cfg = cfg.get("loss", {})
    logger.info("Deriving class weights from training set only...")
    class_weights = compute_class_weights_from_manifest(
        train_manifest_csv=train_manifest,
        dataset_dir=dataset_dir,
        num_classes=6,
        weight_power=loss_cfg.get("weight_power", 0.5),
        max_weight=loss_cfg.get("max_weight", 75.0),
    ).to(device)

    loss_fn = CombinedDiceAndCrossEntropyLoss(
        class_weights=class_weights,
        alpha_ce=loss_cfg.get("alpha_ce", 1.0),
        beta_dice=loss_cfg.get("beta_dice", 2.0),
        focal_gamma=loss_cfg.get("focal_gamma", 1.5),
        tversky_alpha=loss_cfg.get("tversky_alpha", 0.3),
        tversky_beta=loss_cfg.get("tversky_beta", 0.7),
        num_classes=6,
        include_background_dice=loss_cfg.get("include_background_dice", False),
    )

    # 5. Optimizer & Scheduler
    train_params = cfg.get("training", {})
    lr = float(train_params.get("learning_rate", 3e-4))
    weight_decay = float(train_params.get("weight_decay", 1e-4))
    num_epochs = 1 if args.smoke_test else (args.epochs or train_params.get("num_epochs", 45))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=float(train_params.get("min_lr", 1e-5)),
    )

    # 6. Trainer Execution
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device,
        checkpoint_dir=cfg["output"].get("checkpoint_dir", "outputs/checkpoints"),
        metrics_dir=cfg["output"].get("metrics_dir", "outputs/metrics"),
        use_amp=train_params.get("use_amp", True),
        grad_clip_norm=train_params.get("grad_clip_norm", 1.0),
        early_stopping_patience=train_params.get("early_stopping_patience", 15),
    )

    smoke_test_batches = 20 if args.smoke_test else None
    logger.info(f"Starting training run ({num_epochs} epoch(s), max_batches={smoke_test_batches})...")
    history = trainer.train(num_epochs=num_epochs, smoke_test_batches=smoke_test_batches)

    # 7. Post-training Evaluation & Plots
    logger.info("Plotting training loss and Dice curves...")
    history_csv = os.path.join(cfg["output"].get("metrics_dir", "outputs/metrics"), "training_history.csv")
    curve_fig = os.path.join(cfg["output"].get("figures_dir", "outputs/figures"), "training_curves.png")
    plot_training_curves(history_csv, save_path=curve_fig)

    # Load best checkpoint for full validation evaluation
    best_ckpt = os.path.join(cfg["output"].get("checkpoint_dir", "outputs/checkpoints"), "best_model.pth")
    if os.path.exists(best_ckpt):
        logger.info(f"Loading best checkpoint from {best_ckpt} for final validation evaluation...")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    logger.info("Running validation partition evaluation...")
    eval_metrics = evaluate_model_on_dataset(
        model=model,
        val_loader=val_loader,
        device=device,
        output_dir=cfg["output"].get("metrics_dir", "outputs/metrics"),
        figure_dir=cfg["output"].get("figures_dir", "outputs/figures"),
        max_batches=smoke_test_batches,
    )

    logger.info("Training and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
