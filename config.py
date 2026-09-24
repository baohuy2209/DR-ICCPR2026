"""Typed Python dataclass configurations for drdg experiments.

Strictly avoids YAML, JSON, or Hydra configuration files in compliance
with project specifications. Exposes clean CLI override mappings.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from drdg.paths import (
    CHECKPOINTS_DIR,
    LODO_FOLDS_DIR,
    LODO_RESULTS_DIR,
    MASTER_METADATA_CSV,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


@dataclass
class DataConfig:
    """Configuration for data ingestion, preprocessing, and batch loading."""

    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    metadata_csv: Path = MASTER_METADATA_CSV
    lodo_folds_dir: Path = LODO_FOLDS_DIR

    # Resolutions
    global_image_size: int = 512
    native_crop_quality: int = 95

    # Dataloader parameters
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    # Registered cohorts
    registered_cohorts: List[str] = field(
        default_factory=lambda: [
            "aptos",
            "ddr",
            "deepdrid",
            "idrid",
            "messidor2",
            "eyepacs",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        for k, v in res.items():
            if isinstance(v, Path):
                res[k] = str(v)
        return res


@dataclass
class LocalMILConfig:
    """Configuration for native-resolution sliding-window patch extraction and MIL."""

    patch_size: int = 224
    stride: int = 112
    fov_threshold: float = 0.35
    max_patches_train: int = 24
    max_patches_eval: int = 48
    feature_dim: int = 512
    gated_attention: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Configuration for multi-scale dual-branch architectures and output heads."""

    variant: str = "v3"  # 'v0', 'v1', 'v2', 'v3'
    head_type: str = "clm_qwk"  # 'softmax_ce', 'softmax_qwk', 'coral', 'corn', 'clm_qwk'
    num_classes: int = 5
    backbone_name: str = "resnet50"
    pretrained: bool = True

    # Global context parameters
    global_feature_dim: int = 512
    use_cbam: bool = True
    use_non_local: bool = True
    use_quadrant_tokens: bool = True

    # Local MIL stream parameters
    local_mil: LocalMILConfig = field(default_factory=LocalMILConfig)

    # Fusion dimension (Global 512 + Local 512 = 1024)
    fusion_dim: int = 1024
    dropout: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["local_mil"] = self.local_mil.to_dict()
        return res


@dataclass
class TrainingConfig:
    """Hyperparameters and execution parameters for model training."""

    epochs: int = 30
    lr: float = 1e-4
    backbone_lr_ratio: float = 0.1
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0

    # Learning rate schedule
    scheduler_type: str = "cosine"  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Loss weights
    ce_loss_weight: float = 1.0
    qwk_loss_weight: float = 0.5
    ordinal_loss_weight: float = 1.0

    # Early stopping and checkpointing
    early_stopping_patience: int = 7
    monitor_metric: str = "qwk"  # 'qwk', 'val_loss'

    # Acceleration
    amp_mode: str = "auto"  # 'auto', 'fp16', 'bf16', 'none'
    device: str = "auto"  # 'auto', 'cuda', 'cpu'

    checkpoints_dir: Path = CHECKPOINTS_DIR
    resume_checkpoint: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        for k, v in res.items():
            if isinstance(v, Path):
                res[k] = str(v)
        return res


@dataclass
class LODOConfig:
    """Configuration for six-fold Leave-One-Dataset-Out cross-domain evaluation."""

    variant: str = "v3"
    held_out_dataset: Optional[str] = "aptos"  # 'aptos', 'ddr', 'deepdrid', 'idrid', 'messidor2', 'eyepacs'
    all_folds: bool = False
    splits_dir: Path = LODO_FOLDS_DIR
    results_dir: Path = LODO_RESULTS_DIR
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["data"] = self.data.to_dict()
        res["model"] = self.model.to_dict()
        res["training"] = self.training.to_dict()
        for k, v in res.items():
            if isinstance(v, Path):
                res[k] = str(v)
        return res


@dataclass
class XAIConfig:
    """Configuration for explainable AI evaluation against IDRiD pixel lesion masks."""

    target_layer_name: str = "layer4"
    pointing_game_tolerance_pixels: int = 15
    eval_metrics: List[str] = field(
        default_factory=lambda: ["pointing_game", "area_matched_recall", "pixel_auroc"]
    )
    results_dir: Path = RESULTS_DIR / "xai"

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["results_dir"] = str(self.results_dir)
        return res
