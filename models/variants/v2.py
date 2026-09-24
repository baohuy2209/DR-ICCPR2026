"""Variant 2 (V2): Global Context + Local MIL Dual-Branch Architecture.

Ablation configuration:
    - Global Branch: ResNet-50 + Non-Local Block + Quadrant Tokens + CBAM
    - Local Branch: Enabled (EfficientNet-B0 + Gated Attention Dual Pooling)
    - Output Head: Softmax Categorical Cross-Entropy (Softmax_CE)
"""

from drdg.models.full_model import DRFullModel, DualBranchMILModel


def build_v2_model(
    pretrained: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Constructs the V2 model (Full dual-branch backbone + Softmax CE head)."""
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=True,
        use_dual_pooling=True,
        use_local_branch=True,
        use_nonlocal=True,
        use_quadrant_tokens=True,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate,
    )
    return DRFullModel(
        backbone=backbone,
        head_type="Softmax_CE",
        num_classes=num_classes,
    )


class V2GlobalContextModel(DRFullModel):
    """Class wrapper for Variant 2 Dual-Branch model."""

    def __init__(
        self,
        pretrained: bool = True,
        fusion_dim: int = 512,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
    ) -> None:
        model = build_v2_model(
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
