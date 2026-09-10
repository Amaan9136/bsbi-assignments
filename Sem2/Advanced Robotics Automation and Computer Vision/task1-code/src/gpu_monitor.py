"""
gpu_monitor.py

Polls GPU temperature via nvidia-smi so training can pause when the GPU
gets too hot (useful for a local/home GPU without proper cooling) and
resume automatically once it cools down. Safe to import and call on any
machine: if nvidia-smi is not available (no NVIDIA GPU, Kaggle CPU
session, etc.) every function becomes a harmless no-op instead of
raising, so the same code path works on Kaggle and locally without
branching in train.py.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import List, Optional

from .logging_setup import get_logger

logger = get_logger(__name__)

_NVIDIA_SMI_AVAILABLE: Optional[bool] = None


def nvidia_smi_available() -> bool:
    global _NVIDIA_SMI_AVAILABLE
    if _NVIDIA_SMI_AVAILABLE is None:
        _NVIDIA_SMI_AVAILABLE = shutil.which("nvidia-smi") is not None
        if not _NVIDIA_SMI_AVAILABLE:
            logger.info("nvidia-smi not found; GPU temperature throttling disabled.")
    return _NVIDIA_SMI_AVAILABLE


def get_gpu_temperatures() -> List[int]:
    """Returns the current temperature in Celsius for every visible GPU,
    as reported by nvidia-smi. Returns an empty list if nvidia-smi is
    unavailable or the call fails for any reason, so callers never need
    to handle exceptions from this function."""

    if not nvidia_smi_available():
        return []

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return [int(line.strip()) for line in output.decode().splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("Could not read GPU temperature: %s", exc)
        return []


def get_max_gpu_temperature() -> Optional[int]:
    temps = get_gpu_temperatures()
    return max(temps) if temps else None


def wait_for_cooldown(
    high_temp_c: int,
    resume_temp_c: int,
    poll_seconds: int = 15,
    max_wait_seconds: Optional[int] = None,
) -> float:
    """Blocks while the hottest visible GPU is at or above high_temp_c,
    polling every poll_seconds, and returns once it drops to or below
    resume_temp_c. Returns the number of seconds spent waiting. If
    nvidia-smi is unavailable this returns immediately (0.0), so it is
    always safe to call unconditionally between epochs/batches."""

    if not nvidia_smi_available():
        return 0.0

    current = get_max_gpu_temperature()
    if current is None or current < high_temp_c:
        return 0.0

    logger.info(
        "GPU temperature %d\u00b0C reached threshold %d\u00b0C; pausing training until it drops to %d\u00b0C.",
        current, high_temp_c, resume_temp_c,
    )
    waited = 0.0
    while True:
        time.sleep(poll_seconds)
        waited += poll_seconds
        current = get_max_gpu_temperature()
        if current is None:
            logger.warning("Lost GPU temperature reading while cooling down; resuming training.")
            break
        logger.info("Cooling down: GPU at %d\u00b0C (resume at %d\u00b0C)...", current, resume_temp_c)
        if current <= resume_temp_c:
            logger.info("GPU cooled to %d\u00b0C; resuming training.", current)
            break
        if max_wait_seconds is not None and waited >= max_wait_seconds:
            logger.warning(
                "Max cooldown wait of %d seconds reached; resuming training anyway.", max_wait_seconds
            )
            break
    return waited