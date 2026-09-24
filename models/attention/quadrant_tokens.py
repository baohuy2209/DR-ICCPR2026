"""Quadrant Token Aggregator for clinical retinal quadrant relational reasoning."""

import torch
import torch.nn as nn


class QuadrantTokenAggregator(nn.Module):
    """Explicit 4-quadrant relational aggregation matching the ICDR 4-2-1 convention.

    Splits the final feature map into 4 spatial quadrants (top-left, top-right,
    bottom-left, bottom-right). Each quadrant is pooled into a token with learned
    quadrant positional embeddings, and a Transformer encoder layer allows cross-quadrant
    relational comparison before aggregation.
    """

    def __init__(self, in_channels: int = 2048, num_heads: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Learned positional embedding for 4 quadrants
        self.quadrant_pos_embed = nn.Parameter(torch.zeros(1, 4, in_channels))
        nn.init.trunc_normal_(self.quadrant_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        quadrants = [
            x[:, :, :h_mid, :w_mid],   # Top-left
            x[:, :, :h_mid, w_mid:],   # Top-right
            x[:, :, h_mid:, :w_mid],   # Bottom-left
            x[:, :, h_mid:, w_mid:],   # Bottom-right
        ]
        tokens = torch.stack([self.pool(q).flatten(1) for q in quadrants], dim=1)  # [B, 4, C]
        tokens = tokens + self.quadrant_pos_embed
        tokens = self.encoder(tokens)
        quadrant_summary = tokens.mean(dim=1)  # [B, C]
        return quadrant_summary
