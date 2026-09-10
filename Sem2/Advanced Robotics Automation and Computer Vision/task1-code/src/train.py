"""
train.py

Training loop and checkpointing for the drone crop vision classifier.
Kept independent of any specific entry point (CLI script or notebook) so
the same run_training function can be called identically from main.py or
from a notebook cell.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import gpu_monitor
from .config import Config
from .logging_setup import get_logger
from .model import unwrap_model

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    elapsed_seconds: float = 0.0
    thermal_pause_seconds: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "best_val_acc": self.best_val_acc,
            "elapsed_seconds": self.elapsed_seconds,
            "thermal_pause_seconds": self.thermal_pause_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "TrainingHistory":
        return cls(
            train_loss=list(payload.get("train_loss", [])),
            train_acc=list(payload.get("train_acc", [])),
            val_loss=list(payload.get("val_loss", [])),
            val_acc=list(payload.get("val_acc", [])),
            best_val_acc=payload.get("best_val_acc", 0.0),
            elapsed_seconds=payload.get("elapsed_seconds", 0.0),
            thermal_pause_seconds=payload.get("thermal_pause_seconds", 0.0),
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    cfg: Optional[Config] = None,
) -> tuple[float, float, float]:
    """Runs one full pass over loader, in training or evaluation mode
    depending on train_mode. Returns (average_loss, accuracy,
    thermal_pause_seconds). During training, if cfg.enable_thermal_throttle
    is set, GPU temperature is checked every cfg.gpu_temp_check_every_n_batches
    batches and training blocks until the GPU cools down, protecting a
    local GPU that overheats during long runs; this check is a no-op on
    machines without nvidia-smi (e.g. most Kaggle sessions)."""

    model.train() if train_mode else model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    thermal_pause_seconds = 0.0

    with torch.set_grad_enabled(train_mode):
        for batch_idx, (images, labels) in enumerate(loader):
            if (
                train_mode
                and cfg is not None
                and cfg.enable_thermal_throttle
                and batch_idx % cfg.gpu_temp_check_every_n_batches == 0
            ):
                thermal_pause_seconds += gpu_monitor.wait_for_cooldown(
                    high_temp_c=cfg.gpu_high_temp_c,
                    resume_temp_c=cfg.gpu_resume_temp_c,
                    poll_seconds=cfg.gpu_temp_poll_seconds,
                )

            images, labels = images.to(device), labels.to(device)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return running_loss / total_samples, running_correct / total_samples, thermal_pause_seconds


def save_training_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    history: TrainingHistory,
    best_weights: dict,
    next_epoch: int,
    cfg: Config,
) -> None:
    """Saves everything needed to resume training after an interruption
    (Kaggle disconnect, manual stop, crash) partway through a run:
    model/optimizer/scheduler state, training history so far, the best
    weights seen, and which epoch to resume from. State dicts are always
    saved unwrapped (no 'module.' prefix) so a checkpoint saved under
    DataParallel on Kaggle's dual-GPU sessions can still be resumed on a
    single local GPU, and vice versa. Written atomically (temp file then
    rename) so a disconnect mid-save cannot corrupt the checkpoint."""

    payload = {
        "next_epoch": next_epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_weights": best_weights,
        "history": history.as_dict(),
    }
    tmp_path = str(cfg.checkpoint_path) + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cfg.checkpoint_path)
    logger.info("Training checkpoint saved to %s (resume at epoch %d)", cfg.checkpoint_path, next_epoch + 1)


def load_training_checkpoint(cfg: Config) -> Optional[dict]:
    if not cfg.resume_from_checkpoint or not cfg.checkpoint_path.exists():
        return None
    logger.info("Found existing checkpoint at %s; will resume training.", cfg.checkpoint_path)
    return torch.load(cfg.checkpoint_path, map_location="cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
) -> tuple[nn.Module, TrainingHistory]:
    """Fine-tunes model for cfg.num_epochs epochs, tracking training and
    validation loss/accuracy each epoch. Restores the weights from the
    epoch with the highest validation accuracy before returning, since
    validation performance does not necessarily improve monotonically
    with additional training.

    A checkpoint is written to cfg.checkpoint_path after every epoch. If
    cfg.resume_from_checkpoint is True and a checkpoint already exists in
    cfg.output_dir, training resumes from the epoch after the last one
    completed rather than starting over, so a Kaggle disconnect or a
    manual stop only costs the epoch in progress, not the whole run. Once
    training reaches cfg.num_epochs the checkpoint file is left in place;
    call src.train.clear_checkpoint(cfg) to start a fresh run instead of
    resuming."""

    device = cfg.device
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    history = TrainingHistory()
    best_weights = copy.deepcopy(unwrap_model(model).state_dict())
    start_epoch = 0

    checkpoint = load_training_checkpoint(cfg)
    if checkpoint is not None:
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_weights = checkpoint["best_weights"]
        history = TrainingHistory.from_dict(checkpoint["history"])
        start_epoch = checkpoint["next_epoch"]
        logger.info(
            "Resuming training from epoch %d/%d (best val accuracy so far: %.4f)",
            start_epoch + 1, cfg.num_epochs, history.best_val_acc,
        )

    if start_epoch >= cfg.num_epochs:
        logger.info("Checkpoint already covers all %d epochs; nothing left to train.", cfg.num_epochs)
        unwrap_model(model).load_state_dict(best_weights)
        return model, history

    start_time = time.time()
    for epoch in range(start_epoch, cfg.num_epochs):
        train_loss, train_acc, train_pause = run_epoch(
            model, train_loader, criterion, optimizer, device, True, cfg
        )
        val_loss, val_acc, val_pause = run_epoch(
            model, val_loader, criterion, optimizer, device, False, cfg
        )
        scheduler.step(val_loss)
        history.thermal_pause_seconds += train_pause + val_pause

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            best_weights = copy.deepcopy(unwrap_model(model).state_dict())

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch + 1, cfg.num_epochs, train_loss, train_acc, val_loss, val_acc,
        )

        save_training_checkpoint(model, optimizer, scheduler, history, best_weights, epoch + 1, cfg)

    history.elapsed_seconds = time.time() - start_time
    logger.info(
        "Training complete in %.1f minutes (%.1f minutes paused for GPU cooldown). Best val accuracy: %.4f",
        history.elapsed_seconds / 60, history.thermal_pause_seconds / 60, history.best_val_acc,
    )

    unwrap_model(model).load_state_dict(best_weights)
    return model, history


def clear_checkpoint(cfg: Config) -> None:
    """Deletes any existing training checkpoint so the next call to
    train_model starts a fresh run instead of resuming. Call this when
    you deliberately want to restart training from scratch."""
    if cfg.checkpoint_path.exists():
        cfg.checkpoint_path.unlink()
        logger.info("Cleared training checkpoint at %s", cfg.checkpoint_path)


def save_checkpoint(model: nn.Module, label_names: List[str], cfg: Config) -> str:
    payload = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "label_names": label_names,
        "image_size": cfg.image_size,
    }
    torch.save(payload, cfg.model_path)
    logger.info("Model saved to %s", cfg.model_path)
    return str(cfg.model_path)