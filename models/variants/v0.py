"""Variant 0 (V0): Global Context Baseline.

Ablation configuration:
    - Global Branch: ResNet-50 standard (no Non-Local, no CBAM, no Quadrant Tokens)
    - Local Branch: Disabled (no MIL patch stream)
    - Output Head: Softmax Categorical Cross-Entropy (Softmax_CE)
"""

from typing import Optional

from drdg.models.full_model import DRFullModel, DualBranchMILModel


def build_v0_model(
    pretrained: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Constructs the V0 baseline model (Global ResNet-50 + Softmax CE)."""
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=False,
        use_dual_pooling=False,
        use_local_branch=False,
        use_nonlocal=False,
        use_quadrant_tokens=False,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate,
    )
    return DRFullModel(
        backbone=backbone,
        head_type="Softmax_CE",
        num_classes=num_classes,
    )


class V0BaselineModel(DRFullModel):
    """Class wrapper for Variant 0 baseline model."""

    def __init__(
        self,
        pretrained: bool = True,
        fusion_dim: int = 512,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
    ) -> None:
        model = build_v0_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        super().__init__(
            backbone=model.backbone,
            head_type="Softmax_CE",
            num_classes=num_classes,
        )
