"""Atomic checkpoint saving and restoration helpers for PyTorch models."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    """Saves training checkpoint state atomically.

    Args:
        state: State dictionary containing model weights, optimizer, epoch, etc.
        checkpoint_path: Target path for the checkpoint (e.g. model_last.pth).
        is_best: If True, also writes a copy to best_path.
        best_path: Target path for the best model (e.g. model_best.pth).
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    tmp_path = f"{checkpoint_path}.tmp"

    torch.save(state, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    if is_best and best_path:
        os.makedirs(os.path.dirname(os.path.abspath(best_path)), exist_ok=True)
        tmp_best = f"{best_path}.tmp"
        torch.save(state, tmp_best)
        os.replace(tmp_best, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Loads checkpoint state into model, optimizer, scaler, and scheduler.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: PyTorch model module to load weights into.
        optimizer: Optional optimizer to restore state.
        scaler: Optional GradScaler to restore state.
        scheduler: Optional learning rate scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        The raw checkpoint dictionary containing epoch, metrics, and history.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path} (mapped to {device})")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
