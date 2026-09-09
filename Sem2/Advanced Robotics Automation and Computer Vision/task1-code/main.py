#!/usr/bin/env python3
"""
main.py

Command line entry point for the drone crop vision pipeline. Intended for
use in VS Code or any terminal: running

    python main.py --epochs 8 --batch-size 32

executes the full pipeline (data loading, training, evaluation, figure
generation, checkpoint saving) end to end. Every stage is implemented in
src/ and imported here rather than duplicated, so this script and the
companion notebook (notebook/Task1_Drone_Crop_Disease_Classification.ipynb)
stay in sync by construction: both call the same functions.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import Config, set_seed
from src.data import build_dataloaders, load_plantvillage
from src.evaluate import evaluate_model, top_confusions
from src.inference import sample_predictions
from src.logging_setup import get_logger
from src.model import build_model, count_parameters
from src.plotting import (
    plot_augmented_samples,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_inference_samples,
    plot_training_curves,
)
from src.train import save_checkpoint, train_model

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the drone crop vision model.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip figure generation (useful for quick smoke tests or CI).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    set_seed(cfg.seed)
    logger.info("Using device: %s", cfg.device)

    # Data
    train_loader, val_loader, label_names = build_dataloaders(cfg)
    if not args.skip_plots:
        raw_train, _, _ = load_plantvillage(cfg)
        plot_class_distribution(raw_train, label_names, cfg)
        plot_augmented_samples(train_loader, label_names, cfg)

    # Model
    model = build_model(num_classes=len(label_names))
    param_counts = count_parameters(model)
    logger.info("Model parameters: total=%d trainable=%d", param_counts["total"], param_counts["trainable"])

    # Train
    model, history = train_model(model, train_loader, val_loader, cfg)
    if not args.skip_plots:
        plot_training_curves(history, cfg)

    # Evaluate
    result = evaluate_model(model, val_loader, label_names, cfg.device)
    print(result.report_text)
    for true_label, pred_label, count in top_confusions(result, label_names):
        logger.info("Confusion: true=%s predicted=%s count=%d", true_label, pred_label, count)
    if not args.skip_plots:
        plot_confusion_matrix(result, label_names, cfg)

    # Inference samples
    if not args.skip_plots:
        val_dataset = val_loader.dataset
        predictions = sample_predictions(model, val_dataset, label_names, cfg, cfg.device)
        plot_inference_samples(predictions, cfg)

    # Save checkpoint and a machine-readable results summary for the report
    save_checkpoint(model, label_names, cfg)
    summary = {
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "best_val_acc_during_training": history.best_val_acc,
        "training_time_minutes": history.elapsed_seconds / 60,
        "epochs_run": cfg.num_epochs,
        "total_parameters": param_counts["total"],
    }
    summary_path = cfg.output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
