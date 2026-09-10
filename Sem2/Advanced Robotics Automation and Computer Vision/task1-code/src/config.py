"""
config.py

Central configuration for the drone crop vision pipeline. Using a single
dataclass instead of scattered module-level constants means every stage
(data, model, training, evaluation) receives its parameters from one
explicit object, which can be constructed with defaults, overridden from
CLI arguments, or overridden from a notebook cell without touching any
other file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Dataset
    dataset_name: str = "mohanty/PlantVillage"
    dataset_config: str = "default"
    val_split: float = 0.2

    # Image / augmentation
    image_size: int = 224
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    rotation_degrees: float = 25.0
    affine_translate: float = 0.1
    affine_scale_min: float = 0.8
    affine_scale_max: float = 1.2
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    perspective_distortion: float = 0.2
    perspective_prob: float = 0.3
    gaussian_blur_kernel: int = 3

    # DataLoader
    batch_size: int = 32
    num_workers: int = 2

    # Model / training
    num_epochs: int = 8
    learning_rate: float = 3e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2

    # Paths
    output_dir: Path = Path("outputs")
    model_filename: str = "mobilenetv2_plantvillage_drone.pth"
    checkpoint_filename: str = "training_checkpoint.pth"

    # Multi-GPU
    use_data_parallel: bool = True

    # GPU thermal throttling (local GPU protection; ignored if nvidia-smi
    # is unavailable, e.g. on a Kaggle CPU session or non-NVIDIA machine)
    enable_thermal_throttle: bool = True
    gpu_high_temp_c: int = 75
    gpu_resume_temp_c: int = 50
    gpu_temp_poll_seconds: int = 15
    gpu_temp_check_every_n_batches: int = 50

    # Resume from a previous run
    resume_from_checkpoint: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def num_gpus(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @property
    def model_path(self) -> Path:
        return self.output_dir / self.model_filename

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch (including CUDA) RNGs for
    reproducible runs. Should be called once at the start of any script
    or notebook session before data loading or model initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)