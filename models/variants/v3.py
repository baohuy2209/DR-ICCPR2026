"""Variant 3 (V3): Full Multi-Scale MIL with Consistent Ordinal Regression (CLM-QWK).

Ablation configuration:
    - Global Branch: ResNet-50 + Non-Local Block + Quadrant Tokens + CBAM
    - Local Branch: Enabled (EfficientNet-B0 + Gated Attention Dual Pooling)
    - Output Head: Cumulative Link Model with clog-log link and hybrid QWK loss (CLM_QWK)
"""

from drdg.models.full_model import DRFullModel, DualBranchMILModel


def build_v3_model(
    pretrained: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 0.4,
    dropout_rate: float = 0.3,
) -> DRFullModel:
    """Constructs the V3 model (Full dual-branch backbone + CLM-QWK ordinal head)."""
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
        head_type="CLM_QWK",
        num_classes=num_classes,
        qwk_weight=qwk_weight,
    )


class V3OrdinalCLMModel(DRFullModel):
    """Class wrapper for Variant 3 Ordinal CLM model."""

    def __init__(
        self,
        pretrained: bool = True,
        fusion_dim: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 0.4,
        dropout_rate: float = 0.3,
    ) -> None:
        model = build_v3_model(
            pretrained=pretrained,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            qwk_weight=qwk_weight,
            dropout_rate=dropout_rate,
        )
        super().__init__(
            backbone=model.backbone,
            head_type="CLM_QWK",
            num_classes=num_classes,
            qwk_weight=qwk_weight,
        )
