"""Centralized repository path resolution utilities for drdg."""

import os
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # This file is located at <repo_root>/src/drdg/paths.py
    current = Path(__file__).resolve()
    # Go up 3 levels: paths.py -> drdg -> src -> repo_root
    return current.parent.parent.parent


REPO_ROOT = get_repo_root()

# Standard data paths
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = Path(os.environ.get("DATA_RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = Path(os.environ.get("DATA_PROCESSED_DIR", DATA_DIR / "processed"))
RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR
HYBRID_DIR = RAW_DATA_DIR / "hybrid"
NATIVE_CROPPED_DIR = HYBRID_DIR / "native_cropped"

# Master metadata and LODO folds
MASTER_METADATA_CSV = PROCESSED_DATA_DIR / "master_fundus_metadata.csv"
LODO_FOLDS_DIR = PROCESSED_DATA_DIR / "lodo_folds"
LODO_SUMMARY_CSV = LODO_FOLDS_DIR / "lodo_splits_summary.csv"

# Model checkpoints and results
CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", REPO_ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))
LODO_RESULTS_DIR = RESULTS_DIR / "lodo"
HEAD_ABLATION_RESULTS_DIR = RESULTS_DIR / "head_ablation"
XAI_RESULTS_DIR = RESULTS_DIR / "xai"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_directories_exist() -> None:
    """Ensure standard runtime directories exist."""
    for p in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        LODO_FOLDS_DIR,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
        LODO_RESULTS_DIR,
        HEAD_ABLATION_RESULTS_DIR,
        XAI_RESULTS_DIR,
        FIGURES_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
