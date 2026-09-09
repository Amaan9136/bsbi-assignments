"""
plotting.py

Matplotlib visualisation helpers, kept separate from the core pipeline
logic so that data/model/train/evaluate remain importable and testable
in headless environments without a display backend or matplotlib
installed as a hard requirement of the core package.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; VS Code/Jupyter/Colab all
# override this automatically when an interactive backend is available,
# but this default keeps `python main.py` runnable over SSH or in CI.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .config import Config
from .data import denormalise
from .evaluate import EvaluationResult
from .logging_setup import get_logger

logger = get_logger(__name__)


def _save(fig, cfg: Config, filename: str) -> Path:
    out_path = cfg.output_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", out_path)
    return out_path


def plot_class_distribution(raw_dataset, label_names: List[str], cfg: Config) -> Path:
    counts = Counter(raw_dataset["label"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(label_names)), [counts.get(i, 0) for i in range(len(label_names))])
    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_title("PlantVillage class distribution")
    fig.tight_layout()
    return _save(fig, cfg, "class_distribution.png")


def plot_augmented_samples(loader, label_names: List[str], cfg: Config, n: int = 5) -> Path:
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for i in range(n):
        img = denormalise(images[i], cfg).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(label_names[labels[i]], fontsize=8)
        axes[i].axis("off")
    fig.tight_layout()
    return _save(fig, cfg, "augmented_samples.png")


def plot_training_curves(history, cfg: Config) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.train_loss, label="Train")
    axes[0].plot(history.val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.train_acc, label="Train")
    axes[1].plot(history.val_acc, label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    return _save(fig, cfg, "training_curves.png")


def plot_confusion_matrix(result: EvaluationResult, label_names: List[str], cfg: Config) -> Path:
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=result.confusion, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title("Confusion matrix (validation set)")
    fig.tight_layout()
    np.save(cfg.output_dir / "confusion_matrix.npy", result.confusion)
    return _save(fig, cfg, "confusion_matrix.png")


def plot_inference_samples(prediction_results: List[dict], cfg: Config) -> Path:
    n = len(prediction_results)
    cols = max(1, n // 2)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    axes = np.array(axes).flatten()

    for ax, res in zip(axes, prediction_results):
        img = denormalise(res["image_tensor"], cfg).permute(1, 2, 0).numpy()
        ax.imshow(img)
        colour = "green" if res["correct"] else "red"
        ax.set_title(
            f"Pred: {res['predicted_label']}\n({res['confidence']*100:.1f}%)\n"
            f"True: {res['true_label']}",
            fontsize=7, color=colour,
        )
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, cfg, "inference_samples.png")
