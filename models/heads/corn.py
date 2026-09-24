"""Conditional Ordinal Regression Neural Network (CORN) head.

Reference:
    Shi, Cao, & Raschka (2021). Deep neural networks for rank-consistent ordinal
    regression based on conditional probabilities. Pattern Recognition Letters, 152, 1-8.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.ordinal import corn_loss


class CORNHead(nn.Module):
    """Conditional Ordinal Regression Neural Network (CORN) output head.

    Features:
        - Models forward conditional transition probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
        - Chain rule multiplication: P(y > k) = prod_{j=0}^k P(y > j | y >= j).
        - Non-increasing cumulative probabilities guaranteed by cumulative product.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes conditional logits, cumulative probabilities, and discrete predictions."""
        cond_logits = self.fc(x)  # (B, num_thresholds)
        cond_probs = torch.sigmoid(cond_logits)  # P(y > k | y >= k)
        cum_probs = torch.cumprod(cond_probs, dim=1)  # P(y > k)

        preds = torch.sum(cum_probs > 0.5, dim=1).long()

        b = x.size(0)
        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)

        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "cum_probs": cum_probs,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes conditional binary cross-entropy loss."""
        return corn_loss(outputs["cond_logits"], targets, num_thresholds=self.num_thresholds)
