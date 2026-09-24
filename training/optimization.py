"""Optimizer and learning rate scheduler builders."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    separate_decay: bool = True,
) -> Optimizer:
    """Builds optimizer with optional weight decay separation for norm and bias parameters.

    Args:
        model: PyTorch model.
        optimizer_name: 'adamw' or 'sgd'.
        lr: Base learning rate.
        weight_decay: L2 regularization strength.
        separate_decay: If True, avoids applying weight decay to 1D params (bias, LayerNorm, BatchNorm).

    Returns:
        Configured PyTorch optimizer.
    """
    if separate_decay:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    opt_name = optimizer_name.strip().lower()
    if opt_name == "adamw":
        return AdamW(param_groups, lr=lr)
    if opt_name == "sgd":
        return SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: 'adamw', 'sgd'.")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = "cosine",
    epochs: int = 50,
    min_lr: float = 1e-6,
    patience: int = 4,
    factor: float = 0.5,
) -> Optional[Any]:
    """Builds learning rate scheduler.

    Args:
        optimizer: Configured PyTorch optimizer.
        scheduler_name: 'cosine', 'plateau', or 'none'.
        epochs: Total training epochs (used as T_max for cosine).
        min_lr: Lower bound on learning rate.
        patience: Epochs with no improvement before reducing LR (for plateau).
        factor: Multiplicative factor for LR reduction (for plateau).

    Returns:
        PyTorch learning rate scheduler or None.
    """
    sched_norm = scheduler_name.strip().lower()
    if sched_norm in ("cosine", "cosine_annealing"):
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if sched_norm in ("plateau", "reduce_on_plateau"):
        return ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience, min_lr=min_lr)
    if sched_norm in ("none", "constant"):
        return None

    raise ValueError(f"Unknown scheduler '{scheduler_name}'. Supported: 'cosine', 'plateau', 'none'.")


def update_qwk_warmup(model: nn.Module, epoch: int, warmup_epochs: int) -> float:
    """Updates the dynamic lambda weight for continuous QWK loss during warmup.

    Args:
        model: PyTorch model module (may have .head attribute).
        epoch: Current 1-based training epoch.
        warmup_epochs: Total warmup epochs.

    Returns:
        Current active QWK lambda weight.
    """
    head = getattr(model, "head", model)
    if hasattr(head, "set_qwk_weight") and hasattr(head, "target_qwk_weight"):
        progress = min(1.0, epoch / max(1, warmup_epochs))
        active_weight = progress * head.target_qwk_weight
        head.set_qwk_weight(active_weight)
        return active_weight
    return 0.0
