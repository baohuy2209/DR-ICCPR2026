"""Multi-scale dual-branch feature fusion module.

Combines high-level global context representations (e.g., from ResNet-50)
with fine-grained local lesion features (e.g., from EfficientNet-B0 MIL)
using a robust projection MLP with LayerNorm to prevent batch-size-1 instability.
"""

from typing import Optional

import torch
import torch.nn as nn


class DualBranchFusion(nn.Module):
    """Fuses global context and local lesion bag representations into a shared latent space.

    Architecture:
        [f_global, f_local] -> Linear(in_dim, fusion_dim) -> LayerNorm -> Mish
        -> Dropout(p) -> Linear(fusion_dim, fusion_dim) -> LayerNorm

    LayerNorm is used instead of BatchNorm to ensure numerical stability even
    when the batch size is 1 (e.g. at the end of a dataset or during inference).
    """

    def __init__(
        self,
        global_dim: int = 4096,
        local_dim: int = 2560,
        fusion_dim: int = 512,
        use_local_branch: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.fusion_dim = fusion_dim
        self.use_local_branch = use_local_branch

        combined_dim = global_dim + local_dim if use_local_branch else global_dim

        self.projection = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        f_global: torch.Tensor,
        f_local: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Projects and fuses features from global and local branches.

        Args:
            f_global: Tensor of shape (B, global_dim).
            f_local: Optional tensor of shape (B, local_dim). Required if
                use_local_branch is True.

        Returns:
            Latent representation tensor of shape (B, fusion_dim).
        """
        if self.use_local_branch and f_local is not None:
            f_fused = torch.cat([f_global, f_local], dim=1)
        else:
            f_fused = f_global

        return self.projection(f_fused)
