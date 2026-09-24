"""Variant 1 (V1): Local Micro-Lesion MIL Architecture.

Ablation configuration:
    - Global Branch: ResNet-50 standard (no Non-Local, no Quadrant Tokens, no CBAM)
    - Local Branch: Enabled (EfficientNet-B0 + Gated Attention Dual Pooling)
    - Output Head: Softmax Categorical Cross-Entropy (Softmax_CE)
"""

from drdg.models.full_model import DRFullModel, DualBranchMILModel


def build_v1_model(
    pretrained: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Constructs the V1 model (Global ResNet-50 + Local EfficientNet MIL + Softmax CE)."""
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=False,
        use_dual_pooling=True,
        use_local_branch=True,
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


class V1LocalMILModel(DRFullModel):
    """Class wrapper for Variant 1 Local MIL model."""

    def __init__(
        self,
        pretrained: bool = True,
        fusion_dim: int = 512,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
    ) -> None:
        model = build_v1_model(
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
