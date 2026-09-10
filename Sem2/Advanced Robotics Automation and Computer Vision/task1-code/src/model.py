"""
model.py

Model construction for the drone crop vision classifier. MobileNetV2 is
used as the backbone, pretrained on ImageNet, with the classifier head
replaced to match the number of PlantVillage classes. MobileNetV2 is
selected for its comparatively low parameter count relative to standard
convolutional architectures, which is the primary justification for its
use given the eventual target of onboard inference on embedded drone
hardware (see report section 3.5 for the full justification).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .logging_setup import get_logger

logger = get_logger(__name__)


def build_model(num_classes: int, pretrained: bool = True, use_data_parallel: bool = True) -> nn.Module:
    """Constructs a MobileNetV2 classifier with a fresh linear head sized
    to num_classes. Setting pretrained=False is supported for unit
    testing without downloading ImageNet weights.

    When use_data_parallel is True and more than one CUDA GPU is visible
    (e.g. a dual-T4 Kaggle session), the model is wrapped in
    nn.DataParallel so batches are split across all visible GPUs
    automatically. On a single-GPU or CPU machine (e.g. a local GPU) this
    wrapping is skipped and the model is returned as-is, so the same call
    works unchanged in both environments."""

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("Detected %d GPUs; wrapping model in nn.DataParallel.", torch.cuda.device_count())
        model = nn.DataParallel(model)

    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Returns the underlying module if model is wrapped in
    nn.DataParallel, otherwise returns model unchanged. Use this before
    accessing architecture-specific attributes (e.g. model.classifier)
    that only exist on the unwrapped module."""
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_checkpoint(path: str, num_classes: int, device: torch.device, use_data_parallel: bool = True) -> nn.Module:
    """Loads a model checkpoint saved by src.train.save_checkpoint.
    Rebuilds the architecture (without downloading pretrained weights,
    since the checkpoint already contains fine-tuned weights) and loads
    the saved state dict. State dicts are always saved without the
    DataParallel 'module.' prefix (see save_checkpoint), so this loads
    correctly whether the checkpoint was trained on one GPU or several,
    and whether it is now being loaded on one GPU, several, or CPU."""
    model = build_model(num_classes=num_classes, pretrained=False, use_data_parallel=use_data_parallel)
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s", path)
    return model