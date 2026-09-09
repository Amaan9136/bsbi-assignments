"""
inference.py

Single-image and batch inference helpers, plus visualisation utilities
for producing the sample prediction grids typically captured as report
screenshots.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .config import Config
from .logging_setup import get_logger

logger = get_logger(__name__)


def predict_image(
    model: nn.Module,
    image: Image.Image,
    label_names: List[str],
    cfg: Config,
    device: torch.device,
) -> Tuple[str, float]:
    """Runs inference on a single PIL image and returns the predicted
    label name and confidence score. Applies the same resize and
    normalisation used for validation data (no augmentation)."""

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

    model.eval()
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return label_names[pred_idx.item()], confidence.item()


def sample_predictions(
    model: nn.Module,
    dataset,
    label_names: List[str],
    cfg: Config,
    device: torch.device,
    n_samples: int = 8,
) -> List[dict]:
    """Runs inference on n_samples random items from a
    PlantVillageTorchDataset (validation split) and returns a list of
    dicts with prediction details, suitable for both logging and
    plotting."""

    model.eval()
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    results = []

    with torch.no_grad():
        for idx in indices:
            image_tensor, true_label = dataset[idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = probs.max(dim=1)

            results.append({
                "index": idx,
                "image_tensor": image_tensor,
                "true_label": label_names[true_label],
                "predicted_label": label_names[pred_idx.item()],
                "confidence": confidence.item(),
                "correct": pred_idx.item() == true_label,
            })

    return results
