"""Cumulative Link Model (CLM) ordinal head with complementary log-log link.

Mathematical Formulation:
    - Latent Severity Score: s(x) = W^T x
    - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4 parameterized
      via softplus increments: theta_k = theta_{k-1} + softplus(alpha_k) + 1e-4
    - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
    - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
    - Rank Probability: P(Y > k) = 1 - P(Y <= k)
    - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
    - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.qwk import ContinuousQWKLoss


class CumulativeLinkModelQWKHead(nn.Module):
    """Cumulative Link Model (CLM) head with softplus cutpoints and clog-log link."""

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0

        self.score_fc = nn.Linear(in_features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)

        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float) -> None:
        """Dynamically updates the lambda weight for continuous QWK loss during warmup."""
        self.current_qwk_weight = float(weight)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        """Constructs strictly monotonic cutpoints using softplus increments."""
        increments = F.softplus(self.cutpoint_increments) + 1e-4
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Complementary log-log cumulative distribution function: F(z) = 1 - exp(-exp(z))."""
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing severity scores, cumulative probabilities, and predictions."""
        b = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)
        cutpoints = self._get_monotonic_cutpoints()

        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        cum_probs = self._cloglog_cdf(z)
        rank_probs = 1.0 - cum_probs

        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)

        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()

        return {
            "severity_score": severity_score,
            "latent_score": severity_score,
            "cutpoints": cutpoints,
            "thetas": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds,
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes hybrid loss: NLL + lambda * Continuous_QWK."""
        probs = outputs["probs"]
        b = targets.size(0)
        eps = 1e-7

        target_probs = probs[torch.arange(b, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))

        if class_weights is not None:
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()

        qwk = self.qwk_loss(probs, targets)
        return nll_loss + self.current_qwk_weight * qwk
