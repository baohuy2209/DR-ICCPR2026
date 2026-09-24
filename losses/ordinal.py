"""Ordinal classification and ranking loss functions.

Includes:
- CORAL loss: Multi-task binary cross-entropy across rank indicators.
- CORN loss: Conditional subset binary cross-entropy across forward transitions.
- Ordinal NLL loss: Negative log-likelihood over reconstructed categorical probabilities.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(
    rank_logits: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 4,
) -> torch.Tensor:
    """Computes binary cross-entropy loss for Consistent Rank Logits (CORAL).

    Args:
        rank_logits: Tensor of shape (B, num_thresholds).
        targets: Integer target tensor of shape (B,).
        num_thresholds: Number of binary cutpoints (num_classes - 1).

    Returns:
        Scalar binary cross-entropy loss.
    """
    batch_size = targets.size(0)
    device = targets.device

    binary_targets = torch.zeros(batch_size, num_thresholds, device=device)
    for k in range(num_thresholds):
        binary_targets[:, k] = (targets > k).float()

    return F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")


def corn_loss(
    cond_logits: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 4,
) -> torch.Tensor:
    """Computes conditional binary cross-entropy loss for CORN.

    Each sub-task k only considers samples where target >= k.

    Args:
        cond_logits: Tensor of shape (B, num_thresholds).
        targets: Integer target tensor of shape (B,).
        num_thresholds: Number of conditional tasks (num_classes - 1).

    Returns:
        Scalar conditional binary cross-entropy loss.
    """
    loss = torch.tensor(0.0, device=targets.device)
    valid_tasks = 0

    for k in range(num_thresholds):
        mask = targets >= k
        if mask.sum() > 0:
            subset_logits = cond_logits[mask, k]
            subset_targets = (targets[mask] > k).float()
            task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
            loss = loss + task_loss
            valid_tasks += 1

    return loss / max(1, valid_tasks)


def ordinal_nll_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes negative log-likelihood on ordinal categorical probabilities.

    Args:
        probs: Categorical probabilities (B, num_classes).
        targets: Integer labels (B,).
        class_weights: Optional per-class weight tensor (num_classes,).
        eps: Minimum probability value for clamping to prevent log(0).

    Returns:
        Scalar NLL loss.
    """
    batch_size = targets.size(0)
    target_probs = probs[torch.arange(batch_size, device=targets.device), targets]
    nll = -torch.log(torch.clamp(target_probs, min=eps))

    if class_weights is not None:
        weights = class_weights[targets]
        return (nll * weights).mean()
    return nll.mean()


class CORALLoss(nn.Module):
    """Module wrapper for CORAL ordinal binary cross-entropy loss."""

    def __init__(self, num_thresholds: int = 4) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

    def forward(self, rank_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return coral_loss(rank_logits, targets, num_thresholds=self.num_thresholds)


class CORNLoss(nn.Module):
    """Module wrapper for CORN conditional binary cross-entropy loss."""

    def __init__(self, num_thresholds: int = 4) -> None:
        super().__init__()
        self.num_thresholds = num_thresholds

    def forward(self, cond_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return corn_loss(cond_logits, targets, num_thresholds=self.num_thresholds)


class OrdinalNLLLoss(nn.Module):
    """Module wrapper for ordinal negative log-likelihood loss."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, eps: float = 1e-7) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return ordinal_nll_loss(probs, targets, class_weights=self.class_weights, eps=self.eps)
