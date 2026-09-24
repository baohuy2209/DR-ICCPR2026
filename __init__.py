"""drdg: Deconstructing Domain Shift in Diabetic Retinopathy Grading.

Multi-Scale MIL with Consistent Ordinal Regression across Heterogeneous Retinal Cohorts.
"""

__version__ = "0.1.0"
__author__ = "ICCPR Research Team"

from drdg.config import (
    DataConfig,
    LocalMILConfig,
    LODOConfig,
    ModelConfig,
    TrainingConfig,
    XAIConfig,
)
from drdg.paths import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    LODO_FOLDS_DIR,
    MASTER_METADATA_CSV,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
)

__all__ = [
    "__version__",
    "DataConfig",
    "LocalMILConfig",
    "ModelConfig",
    "TrainingConfig",
    "LODOConfig",
    "XAIConfig",
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MASTER_METADATA_CSV",
    "LODO_FOLDS_DIR",
    "CHECKPOINTS_DIR",
    "RESULTS_DIR",
]
