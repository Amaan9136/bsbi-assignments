"""
data.py

Dataset loading, splitting and augmentation for the PlantVillage dataset,
adapted with augmentation intended to approximate drone-captured aerial
imaging conditions (viewpoint change, motion blur, variable outdoor
lighting) rather than the clean, close-range conditions of the source
photography.

Loading PlantVillage requires the "color" dataset configuration and uses
the column name "label" (not "labels"); both are handled here, verified
directly against the dataset's Hugging Face card at the time of writing.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config
from .logging_setup import get_logger

logger = get_logger(__name__)


class PlantVillageTorchDataset(Dataset):
    """Wraps a Hugging Face dataset split and applies a torchvision
    transform on access, returning (tensor, label) pairs expected by a
    PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform: transforms.Compose) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]
        return self.transform(image), label


def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """Builds the training (augmented) and validation (clean) transform
    pipelines. Validation data receives resize and normalisation only, so
    reported metrics reflect genuine model performance rather than
    augmented inputs."""

    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomRotation(degrees=cfg.rotation_degrees),
        transforms.RandomAffine(
            degrees=0,
            translate=(cfg.affine_translate, cfg.affine_translate),
            scale=(cfg.affine_scale_min, cfg.affine_scale_max),
        ),
        transforms.ColorJitter(
            brightness=cfg.color_jitter_brightness,
            contrast=cfg.color_jitter_contrast,
            saturation=cfg.color_jitter_saturation,
        ),
        transforms.RandomPerspective(
            distortion_scale=cfg.perspective_distortion, p=cfg.perspective_prob
        ),
        transforms.GaussianBlur(kernel_size=cfg.gaussian_blur_kernel, sigma=(0.1, 1.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])

    return train_transform, val_transform


import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset, ClassLabel, Image as HFImage

def load_plantvillage(cfg: Config):
    logger.info("Loading dataset %s via direct zip extraction", cfg.dataset_name)

    zip_path = hf_hub_download(repo_id=cfg.dataset_name, filename="data.zip", repo_type="dataset")
    extract_dir = Path(cfg.output_dir) / "plantvillage_raw"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    variant = cfg.dataset_config if cfg.dataset_config in ("color", "grayscale", "segmented") else "color"
    root = extract_dir / variant
    if not root.is_dir():
        candidates = list(extract_dir.rglob(variant))
        if not candidates:
            raise ValueError(f"Could not locate '{variant}' folder under {extract_dir}")
        root = candidates[0]

    image_paths, class_names = [], []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append(str(img_path))
                class_names.append(class_dir.name)

    label_names = sorted(set(class_names))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    labels = [label_to_id[c] for c in class_names]

    raw_dataset = Dataset.from_dict({"image": image_paths, "label": labels})
    raw_dataset = raw_dataset.cast_column("image", HFImage())
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=label_names))

    logger.info("Loaded %d images across %d classes", len(raw_dataset), len(label_names))

    split_dataset = raw_dataset.train_test_split(
        test_size=cfg.val_split, seed=cfg.seed, stratify_by_column="label"
    )
    train_raw, val_raw = split_dataset["train"], split_dataset["test"]
    logger.info("Train/val split: %d / %d", len(train_raw), len(val_raw))
    return train_raw, val_raw, label_names

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Convenience entry point combining load_plantvillage, transform
    construction and DataLoader wrapping into a single call."""
    train_raw, val_raw, label_names = load_plantvillage(cfg)
    train_transform, val_transform = build_transforms(cfg)

    train_dataset = PlantVillageTorchDataset(train_raw, train_transform)
    val_dataset = PlantVillageTorchDataset(val_raw, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, label_names


def denormalise(tensor: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Reverses ImageNet normalisation for visualisation purposes."""
    mean = torch.tensor(cfg.imagenet_mean).view(3, 1, 1)
    std = torch.tensor(cfg.imagenet_std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
