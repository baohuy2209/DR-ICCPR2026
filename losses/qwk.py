"""Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

Computes continuous soft confusion matrices to directly optimize clinical
ordinal agreement:
    QWK = 1 - (sum(W * O) / sum(W * E))
    Loss = sum(W * O) / (sum(W * E) + eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousQWKLoss(nn.Module):
    """Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss.

    Minimizing (sum(W * O) / sum(W * E)) directly maximizes Cohen's quadratic
    weighted kappa agreement between soft predicted distributions and targets.
    """

    def __init__(self, num_classes: int = 5, eps: float = 1e-7) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        # Quadratic penalty matrix: W_ij = (i - j)^2 / (C - 1)^2
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

        self.register_buffer("W", w)

    def forward(
        self,
        pred_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation of differentiable QWK loss.

        Args:
            pred_probs: Continuous class probabilities (B, num_classes).
            targets: Integer class indices (B,) or one-hot distributions (B, num_classes).

        Returns:
            Scalar differentiable loss tensor.
        """
        device = pred_probs.device
        n = pred_probs.size(0)
        if n == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()

        # Normalize predicted probabilities to simplex
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)

        # Soft observed confusion matrix: O = (P^T * Y) / N
        observed = torch.matmul(pred_probs.t(), y_one_hot) / (n + self.eps)

        # Expected confusion matrix under chance: E = hist_pred^T * hist_true
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (n + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (n + self.eps)
        expected = torch.matmul(hist_pred.t(), hist_true)

        penalty = self.W.to(device)
        num = torch.sum(penalty * observed)
        den = torch.sum(penalty * expected) + self.eps

        return num / den
