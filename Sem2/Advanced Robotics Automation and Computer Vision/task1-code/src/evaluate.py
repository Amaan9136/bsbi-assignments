"""
evaluate.py

Evaluation metrics and confusion matrix computation for the drone crop
vision classifier. Accuracy alone is not a reliable indicator of
performance on an imbalanced classification problem such as PlantVillage;
precision, recall and F1 are computed per class and macro-averaged, and
the full confusion matrix is returned for downstream visualisation or
reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion: np.ndarray
    report_text: str
    y_true: np.ndarray
    y_pred: np.ndarray


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    label_names: List[str],
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    logger.info(
        "Evaluation: accuracy=%.4f macro_precision=%.4f macro_recall=%.4f macro_f1=%.4f",
        accuracy, precision, recall, f1,
    )

    return EvaluationResult(
        accuracy=accuracy,
        macro_precision=precision,
        macro_recall=recall,
        macro_f1=f1,
        confusion=cm,
        report_text=report_text,
        y_true=y_true,
        y_pred=y_pred,
    )


def top_confusions(result: EvaluationResult, label_names: List[str], n: int = 5) -> List[tuple]:
    """Returns the n largest off-diagonal confusion matrix entries as
    (true_label, predicted_label, count) tuples, sorted descending. This
    is generally more informative for a report discussion than the raw
    matrix, since it directly answers 'what does the model confuse'."""
    cm = result.confusion.copy()
    np.fill_diagonal(cm, 0)

    flat_indices = np.argsort(cm, axis=None)[::-1][:n]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    confusions = []
    for r, c in zip(rows, cols):
        if cm[r, c] > 0:
            confusions.append((label_names[r], label_names[c], int(cm[r, c])))
    return confusions
