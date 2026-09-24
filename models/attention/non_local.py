"""Non-Local Neural Networks block for long-range spatial relational reasoning."""

import torch
import torch.nn as nn


class NonLocalBlock2D(nn.Module):
    """Embedded Gaussian Non-local block (Wang et al., 2018).

    Computes self-attention between all spatial positions in the feature map,
    enabling comparison of lesion signals across distant quadrants.
    The output projection is initialized to zeros so the block begins as an
    identity mapping (residual connection), preventing initial fine-tuning instability.
    """

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // reduction)

        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Zero-initialize the output projection
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW, C']
        phi = self.phi(x).view(B, self.inter_channels, -1)                     # [B, C', HW]
        g = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)          # [B, HW, C']

        attn = torch.softmax(torch.bmm(theta, phi) / (self.inter_channels ** 0.5), dim=-1)  # [B, HW, HW]
        y = torch.bmm(attn, g)                                                  # [B, HW, C']
        y = y.permute(0, 2, 1).contiguous().view(B, self.inter_channels, H, W)
        y = self.out_conv(y)
        return x + y  # Residual connection
