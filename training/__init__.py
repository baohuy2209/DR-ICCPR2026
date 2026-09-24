"""Training engines, optimization builders, and checkpoint management.

Exports:
- DRTrainer: Full multi-scale dual-branch trainer
- DRHeadTrainer: Standardized single-branch head ablation trainer
- build_optimizer, build_scheduler, update_qwk_warmup
- save_checkpoint, load_checkpoint
"""

from drdg.training.checkpoint import load_checkpoint, save_checkpoint
from drdg.training.head_trainer import DRHeadTrainer
from drdg.training.optimization import (
    build_optimizer,
    build_scheduler,
    update_qwk_warmup,
)
from drdg.training.trainer import DRTrainer

__all__ = [
    "DRTrainer",
    "DRHeadTrainer",
    "build_optimizer",
    "build_scheduler",
    "update_qwk_warmup",
    "save_checkpoint",
    "load_checkpoint",
]
