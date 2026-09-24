"""Full end-to-end models for diabetic retinopathy severity grading.

Includes:
- DualBranchMILModel: Backbone extracting global context and local MIL features and fusing them.
- DRFullModel: End-to-end wrapper combining DualBranchMILModel with a pluggable ordinal/classification head.
- ResNet50AblationModel: Single-branch ResNet-50 feature extractor with 512-d projection for controlled head ablations.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

from drdg.models.fusion import DualBranchFusion
from drdg.models.heads.clm import CumulativeLinkModelQWKHead
from drdg.models.heads.coral import CORALHead
from drdg.models.heads.corn import CORNHead
from drdg.models.heads.softmax import SoftmaxCEHead, SoftmaxQWKHead
from drdg.models.streams.global_context import GlobalBranch
from drdg.models.streams.local_mil import LocalMILBranch


class DualBranchMILModel(nn.Module):
    """Unified Dual-Branch Multi-Scale Feature Extraction Backbone.

    Combines Global Context (ResNet-50 + Non-Local + CBAM + Quadrant Tokens, 4096-d)
    and Local Micro-Lesions (MIL EfficientNet-B0 Dual Pooling, 2560-d) into a fused
    latent representation (default 512-d).
    """

    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_dual_pooling: bool = True,
        use_local_branch: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
        fusion_dim: int = 512,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.use_local_branch = use_local_branch
        self.latent_dim = fusion_dim

        self.global_branch = GlobalBranch(
            pretrained=pretrained,
            use_cbam=use_cbam,
            use_nonlocal=use_nonlocal,
            use_quadrant_tokens=use_quadrant_tokens,
        )

        if use_local_branch:
            self.local_mil_branch = LocalMILBranch(
                pretrained=pretrained,
                use_dual_pooling=use_dual_pooling,
            )
            local_dim = self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            local_dim = 0

        self.fusion = DualBranchFusion(
            global_dim=self.global_branch.out_dim,
            local_dim=local_dim,
            fusion_dim=fusion_dim,
            use_local_branch=use_local_branch,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        patch_imgs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extracts and fuses multi-scale features into a latent representation."""
        if local_patches is None and patch_imgs is not None:
            local_patches = patch_imgs
        f_global = self.global_branch(global_img)

        if self.use_local_branch and self.local_mil_branch is not None and local_patches is not None:
            f_local = self.local_mil_branch(local_patches, patch_mask=patch_mask)
            latent = self.fusion(f_global, f_local)
        else:
            latent = self.fusion(f_global, None)

        return latent


class DRFullModel(nn.Module):
    """End-to-end Diabetic Retinopathy Grading Model.

    Pairs a multi-scale backbone (DualBranchMILModel) with any of the candidate
    output classification/ordinal heads.
    """

    def __init__(
        self,
        backbone: DualBranchMILModel,
        head_type: str = "CLM_QWK",
        num_classes: int = 5,
        qwk_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.num_classes = num_classes
        in_features = backbone.latent_dim

        if head_type in ("Softmax_CE", "softmax_ce"):
            self.head: nn.Module = SoftmaxCEHead(in_features=in_features, num_classes=num_classes)
        elif head_type in ("Softmax_QWK", "softmax_qwk"):
            self.head = SoftmaxQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_type in ("CORAL", "coral"):
            self.head = CORALHead(in_features=in_features, num_classes=num_classes)
        elif head_type in ("CORN", "corn"):
            self.head = CORNHead(in_features=in_features, num_classes=num_classes)
        elif head_type in ("CLM_QWK", "clm_qwk", "CLM", "clm"):
            self.head = CumulativeLinkModelQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'. Valid types: Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK.")

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        patch_imgs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward pass through backbone and prediction head."""
        if local_patches is None and patch_imgs is not None:
            local_patches = patch_imgs
        latent = self.backbone(global_img, local_patches=local_patches, patch_mask=patch_mask)
        head_outputs = self.head(latent)
        head_outputs["latent"] = latent

        if return_loss and targets is not None:
            head_outputs["loss"] = self.head.loss_fn(
                head_outputs, targets, class_weights=class_weights
            )

        return head_outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss using the model's attached head."""
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)


class ResNet50AblationModel(nn.Module):
    """Standardized single-branch ResNet-50 Feature Extractor with 512-d Projection.

    Used for controlled head ablation benchmarks (comparing Softmax, CORAL, CORN,
    and CLM on identical representation backbones).
    """

    def __init__(
        self,
        head_type: str = "CLM_QWK",
        pretrained: bool = True,
        num_classes: int = 5,
        qwk_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.num_classes = num_classes
        self.latent_dim = 512

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            base_resnet.conv1, base_resnet.bn1, base_resnet.relu, base_resnet.maxpool
        )
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.2),
        )

        if head_type in ("Softmax_CE", "softmax_ce"):
            self.head: nn.Module = SoftmaxCEHead(in_features=512, num_classes=num_classes)
        elif head_type in ("Softmax_QWK", "softmax_qwk"):
            self.head = SoftmaxQWKHead(in_features=512, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_type in ("CORAL", "coral"):
            self.head = CORALHead(in_features=512, num_classes=num_classes)
        elif head_type in ("CORN", "corn"):
            self.head = CORNHead(in_features=512, num_classes=num_classes)
        elif head_type in ("CLM_QWK", "clm_qwk", "CLM", "clm"):
            self.head = CumulativeLinkModelQWKHead(in_features=512, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-d latent representation from an input image tensor."""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out).flatten(1)
        return self.proj(out)

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass matching the standard model interface."""
        latent = self.extract_features(global_img)
        outputs = self.head(latent)
        outputs["latent"] = latent

        if return_loss and targets is not None:
            outputs["loss"] = self.head.loss_fn(
                outputs, targets, class_weights=class_weights
            )

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes loss using the attached head."""
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)
