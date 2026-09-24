"""Loss functions for diabetic retinopathy severity classification.

Includes:
- ContinuousQWKLoss: Differentiable Quadratic Weighted Kappa approximation
- coral_loss, CORALLoss: Consistent Rank Logits multi-task BCE
- corn_loss, CORNLoss: Conditional Ordinal Regression Neural Network loss
- ordinal_nll_loss, OrdinalNLLLoss: Negative log-likelihood over reconstructed probabilities
"""

from drdg.losses.ordinal import (
    CORALLoss,
    CORNLoss,
    OrdinalNLLLoss,
    coral_loss,
    corn_loss,
    ordinal_nll_loss,
)
from drdg.losses.qwk import ContinuousQWKLoss

__all__ = [
    "ContinuousQWKLoss",
    "coral_loss",
    "corn_loss",
    "ordinal_nll_loss",
    "CORALLoss",
    "CORNLoss",
    "OrdinalNLLLoss",
]
