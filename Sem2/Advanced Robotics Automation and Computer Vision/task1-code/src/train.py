"""
train.py

Training loop and checkpointing for the drone crop vision classifier.
Kept independent of any specific entry point (CLI script or notebook) so
the same run_training function can be called identically from main.py or
from a notebook cell.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import Config
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "best_val_acc": self.best_val_acc,
            "elapsed_seconds": self.elapsed_seconds,
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train_mode: bool,
) -> tuple[float, float]:
    """Runs one full pass over loader, in training or evaluation mode
    depending on train_mode. Returns (average_loss, accuracy)."""

    model.train() if train_mode else model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
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

    return running_loss / total_samples, running_correct / total_samples


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
    with additional training."""

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
    best_weights = copy.deepcopy(model.state_dict())

    start_time = time.time()
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch + 1, cfg.num_epochs, train_loss, train_acc, val_loss, val_acc,
        )

    history.elapsed_seconds = time.time() - start_time
    logger.info(
        "Training complete in %.1f minutes. Best val accuracy: %.4f",
        history.elapsed_seconds / 60, history.best_val_acc,
    )

    model.load_state_dict(best_weights)
    return model, history


def save_checkpoint(model: nn.Module, label_names: List[str], cfg: Config) -> str:
    payload = {
        "model_state_dict": model.state_dict(),
        "label_names": label_names,
        "image_size": cfg.image_size,
    }
    torch.save(payload, cfg.model_path)
    logger.info("Model saved to %s", cfg.model_path)
    return str(cfg.model_path)
