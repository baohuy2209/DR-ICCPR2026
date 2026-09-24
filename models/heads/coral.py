"""Consistent Rank Logits (CORAL) head for ordinal regression.

Reference:
    Cao, Mirjalili, & Raschka (2020). Rank consistent ordinal regression for neural networks.
    Pattern Recognition Letters, 140, 325-331.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.ordinal import coral_loss


class CORALHead(nn.Module):
    """Consistent Rank Logits (CORAL) output head.

    Features:
        - Shared weight vector W across all binary threshold tasks (parallel hyperplanes).
        - Independent learnable threshold biases b_1, b_2, ..., b_{K-1}.
        - Cumulative probability formulation: P(y > k) = sigmoid(W^T x + b_k).
        - Interval probabilities reconstructed via consecutive differences with ReLU.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 5) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))

        # Guide monotonic training by initializing descending thresholds
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes rank logits, cumulative threshold probabilities, and class predictions."""
        base_logit = self.linear(x)  # (B, 1)
        rank_logits = base_logit + self.biases  # (B, num_thresholds)
        rank_probs = torch.sigmoid(rank_logits)  # P(y > k) in [0, 1]

        # Discrete prediction: count exceeded 0.5 thresholds
        preds = torch.sum(rank_probs > 0.5, dim=1).long()

        # Reconstruct interval probabilities: P(y = k) = P(y > k-1) - P(y > k)
        b = x.size(0)
        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)

        return {
            "rank_logits": rank_logits,
            "rank_probs": rank_probs,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes multi-task binary cross-entropy loss across threshold indicators."""
        return coral_loss(outputs["rank_logits"], targets, num_thresholds=self.num_thresholds)
