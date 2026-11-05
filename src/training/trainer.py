"""
Training and Validation Engine for 2.5D Brain CT Hemorrhage Segmentation.

Key Features:
- AdamW optimizer with Cosine Annealing learning rate schedule.
- Mixed precision training with torch.amp for fast GPU execution and low VRAM usage.
- Gradient clipping to prevent gradient explosion.
- Best checkpoint selection strictly by Macro Dice across the 5 hemorrhage classes (excluding background).
- Comprehensive per-epoch logging (loss, per-class Dice, recall, precision, macro Dice).
- Early stopping support.
"""

import os
import time
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import CombinedDiceAndCrossEntropyLoss
from src.training.metrics import SegmentationMetricsMeter, format_metrics_table
from src.data.split import TARGET_NAMES


class EarlyStopping:
    """
    Early Stopping monitor that tracks validation macro Dice.
    Terminates training if no improvement greater than min_delta is observed for patience epochs.
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_epoch = -1

    def step(self, current_score: float, epoch: int) -> bool:
        """
        Returns True if this epoch yielded a new best score, False otherwise.
        """
        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir: str = "outputs/checkpoints",
        metrics_dir: str = "outputs/metrics",
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.metrics_dir = metrics_dir
        self.use_amp = use_amp and (device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="max",
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        self.best_macro_dice = -1.0
        self.best_epoch = -1
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_dice": [],
            "val_macro_iou": [],
            "lr": [],
        }
        for name in TARGET_NAMES:
            self.history[f"val_dice_{name}"] = []
            self.history[f"val_recall_{name}"] = []

    def train_epoch(self, epoch: int, max_batches: Optional[int] = None) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader) if max_batches is None else min(len(self.train_loader), max_batches)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for step, (images, masks, _) in enumerate(pbar):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            self.scaler.scale(loss).backward()

            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int, max_batches: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader) if max_batches is None else min(len(self.val_loader), max_batches)
        metrics_meter = SegmentationMetricsMeter(num_classes=6)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
        for step, (images, masks, _) in enumerate(pbar):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = masks.cpu().numpy()
            metrics_meter.update(preds, targets_np)

        avg_loss = total_loss / max(1, num_batches)
        metrics = metrics_meter.compute()
        metrics["val_loss"] = avg_loss
        return avg_loss, metrics

    def train(self, num_epochs: int, smoke_test_batches: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes full training loop with checkpointing and early stopping.
        If smoke_test_batches is provided, runs a brief validation smoke test.
        """
        print(f"\n[INFO] Commencing training for {num_epochs} epochs on device: {self.device}")
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch, max_batches=smoke_test_batches)
            val_loss, val_metrics = self.validate(epoch, max_batches=smoke_test_batches)
            elapsed = time.time() - t0

            # Learning rate step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            macro_dice = val_metrics["macro_dice"]
            macro_iou = val_metrics["macro_iou"]

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_dice"].append(macro_dice)
            self.history["val_macro_iou"].append(macro_iou)
            self.history["lr"].append(current_lr)

            for name in TARGET_NAMES:
                self.history[f"val_dice_{name}"].append(val_metrics["per_class_dice"][name])
                self.history[f"val_recall_{name}"].append(val_metrics["per_class_recall"][name])

            # Print epoch summary
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro Dice: {macro_dice:.4f} | Val Macro IoU: {macro_iou:.4f} | LR: {current_lr:.2e}"
            )
            print("  Per-class Val Dice: " + ", ".join(
                [f"{name}: {val_metrics['per_class_dice'][name]:.3f}" for name in TARGET_NAMES[1:]]
            ))

            # Early Stopping and Checkpointing on validation Macro Dice
            is_best = self.early_stopping.step(macro_dice, epoch)
            if is_best:
                self.best_macro_dice = macro_dice
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, filename="best_model.pth")
                print(f"  >>> Best Checkpoint Saved! (Macro Dice: {macro_dice:.4f} at epoch {epoch} | Patience: 0/{self.early_stopping.patience})")
            else:
                print(f"  [Patience: {self.early_stopping.counter}/{self.early_stopping.patience} | Best: {self.early_stopping.best_score:.4f} at Ep {self.early_stopping.best_epoch}]")

            # Save latest checkpoint
            self.save_checkpoint(epoch, val_metrics, filename="latest_model.pth")

            # Save metrics history to CSV
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(os.path.join(self.metrics_dir, "training_history.csv"), index=False)

            # Early stopping check
            if self.early_stopping.early_stop:
                print(f"\n[EARLY STOPPING] Triggered! No improvement for {self.early_stopping.patience} epochs.")
                break

        print(f"\n[COMPLETE] Best Validation Macro Dice: {self.best_macro_dice:.4f} achieved at epoch {self.best_epoch}")

        # Automatically restore best model weights
        best_ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[RESTORE] Loaded best checkpoint weights (epoch {self.best_epoch}) into model for evaluation.")

        return self.history

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any], filename: str = "checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_dice": self.best_macro_dice,
            "metrics": metrics,
        }, path)
