"""Softmax classification heads for diabetic retinopathy grading.

Includes:
- SoftmaxCEHead: Baseline unconstrained categorical head with standard Cross-Entropy.
- SoftmaxQWKHead: Multi-task categorical head with CE + continuous QWK loss and expected grade rounding.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.qwk import ContinuousQWKLoss


class SoftmaxCEHead(nn.Module):
    """Standard Categorical Classification Baseline Head.

    Architecture:
        Linear(in_features, num_classes) -> Softmax
    Loss:
        Categorical Cross-Entropy (treats grades as independent nominal classes).
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes categorical logits, probabilities, and class predictions.

        Args:
            x: Latent representation tensor of shape (B, in_features).

        Returns:
            Dict containing 'logits', 'probs', and 'preds'.
        """
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes categorical cross-entropy loss."""
        return F.cross_entropy(outputs["logits"], targets, weight=class_weights)


class SoftmaxQWKHead(nn.Module):
    """Categorical Head with Hybrid Cross-Entropy and Continuous QWK Loss.

    Inference computes the expected continuous severity:
        E[y] = sum_{k=0}^{C-1} k * P(y = k)
    and rounds to the nearest integer grade clamped to [0, C - 1].
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight

        self.fc = nn.Linear(in_features, num_classes)
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes probabilities and expected continuous severity values."""
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)

        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        continuous_pred = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "continuous_preds": continuous_pred,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes composite multitask loss: CE + lambda * Continuous_QWK."""
        ce_loss = F.cross_entropy(outputs["logits"], targets, weight=class_weights)
        qwk = self.qwk_loss(outputs["probs"], targets)
        return ce_loss + self.qwk_weight * qwk
