"""Global Context stream for macroscopic retinal fundus feature representation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

from drdg.models.attention.cbam import CBAM
from drdg.models.attention.non_local import NonLocalBlock2D
from drdg.models.attention.quadrant_tokens import QuadrantTokenAggregator


class GlobalBranch(nn.Module):
    """Global Context Branch for whole-fundus feature extraction.

    Architecture:
      - Input: Whole-fundus image tensor [Batch, 3, 512, 512]
      - Backbone: Pre-trained ResNet-50 (conv1 -> layer4)
      - Attention: NonLocalBlock2D -> CBAM on layer4 feature maps (2048 channels)
      - Pooling: Global Average Pooling (2048-d) concatenated with Quadrant Token Aggregator (2048-d)
      - Output: Global feature vector F_Global of dimension 4096 (if use_quadrant_tokens) else 2048
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)

        # ResNet stem and stages
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.use_nonlocal = use_nonlocal
        if use_nonlocal:
            self.non_local = NonLocalBlock2D(in_channels=2048)

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(in_channels=2048, reduction_ratio=16)

        self.use_quadrant_tokens = use_quadrant_tokens
        if use_quadrant_tokens:
            self.quadrant_agg = QuadrantTokenAggregator(in_channels=2048)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048 + (2048 if use_quadrant_tokens else 0)

        # Cache for Grad-CAM activations
        self.last_feature_map: Optional[torch.Tensor] = None

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map through layer4 with optional attention."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_nonlocal:
            x = self.non_local(x)
        if self.use_cbam:
            x = self.cbam(x)

        self.last_feature_map = x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating F_Global vector [B, out_dim]."""
        feat_map = self.extract_feature_map(x)
        gap_feat = torch.flatten(self.gap(feat_map), 1)

        if self.use_quadrant_tokens:
            quad_feat = self.quadrant_agg(feat_map)
            out = torch.cat([gap_feat, quad_feat], dim=1)
        else:
            out = gap_feat

        return out


# Public canonical alias
GlobalContextStream = GlobalBranch
