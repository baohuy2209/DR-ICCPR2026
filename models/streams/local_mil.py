"""Local Multi-Instance Learning (MIL) stream with Dual Pooling."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class GatedAttentionPool(nn.Module):
    """Gated Attention Mechanism for Multiple Instance Learning (Ilse et al., 2018).

    Learns instance-level attention weights:
        a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k)))
    Enables lesion counting and spatial density aggregation.
    """

    def __init__(self, in_features: int = 1280, hidden_dim: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Patch features [B, K, in_features]
            patch_mask: Optional boolean tensor [B, K] (True = valid foreground patch)
        Returns:
            weighted_out: Bag-level representation [B, in_features]
        """
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_weights(a_v * a_u)  # [B, K, 1]

        if patch_mask is not None:
            mask_3d = patch_mask.bool().unsqueeze(-1)  # [B, K, 1]
            a = a.masked_fill(~mask_3d, -1e4)

        a = torch.softmax(a, dim=1)  # [B, K, 1]
        self.last_attention_weights = a.detach()

        weighted_out = torch.sum(x * a, dim=1)  # [B, in_features]
        return weighted_out


class LocalMILBranch(nn.Module):
    """Local Multi-Instance Learning Branch with Dual Pooling.

    Processes K native-resolution foreground patches:
      - Patch Backbone: Pre-trained EfficientNet-B0
      - Pooling per patch: AdaptiveAvgPool2d -> [B, K, 1280]
      - Dual Aggregation:
          1. 1D Max-Pooling (extreme focal lesion detector) -> [B, 1280]
          2. Gated Attention Pooling (lesion density aggregator) -> [B, 1280]
      - Output: F_Local vector of dimension 2560 (1280 + 1280) if use_dual_pooling else 1280
    """

    def __init__(self, pretrained: bool = True, use_dual_pooling: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b0(weights=weights)

        self.features = base_effnet.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_dim = 1280
        self.use_dual_pooling = use_dual_pooling

        if use_dual_pooling:
            self.gated_pool = GatedAttentionPool(in_features=self.patch_dim, hidden_dim=128)
            self.out_dim = self.patch_dim * 2  # 2560
        else:
            self.gated_pool = None
            self.out_dim = self.patch_dim      # 1280

    def forward(self, patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            patches: Tensor of shape [B, K, 3, H_p, W_p]
            patch_mask: Optional boolean tensor [B, K]
        Returns:
            f_local: Aggregated local feature vector [B, out_dim]
        """
        B, K, C, H, W = patches.shape
        flat_patches = patches.view(B * K, C, H, W)
        patch_feats = self.features(flat_patches)
        patch_feats = self.gap(patch_feats)
        patch_feats = torch.flatten(patch_feats, 1)
        patch_feats = patch_feats.view(B, K, self.patch_dim)

        # 1. 1D Max-Pooling (focal lesions)
        if patch_mask is not None:
            bool_mask = patch_mask.bool()
            masked_feats = patch_feats.masked_fill(~bool_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)

        if self.use_dual_pooling and self.gated_pool is not None:
            # 2. Gated Attention Pooling (lesion density)
            att_pooled = self.gated_pool(patch_feats, patch_mask=patch_mask)
            f_local = torch.cat([max_pooled, att_pooled], dim=1)
        else:
            f_local = max_pooled

        return f_local
