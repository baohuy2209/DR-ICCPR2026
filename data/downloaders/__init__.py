"""Automated Kaggle dataset downloaders for external fundus cohorts.

Exports:
- download_ddr
- download_deepdrid
- download_idrid
- download_idrid_segmentation
- download_messidor2
- download_dataset: unified dispatcher by cohort name
"""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.ddr import download_ddr
from drdg.data.downloaders.deepdrid import download_deepdrid
from drdg.data.downloaders.idrid import download_idrid
from drdg.data.downloaders.idrid_segmentation import download_idrid_segmentation
from drdg.data.downloaders.messidor2 import download_messidor2

DOWNLOADER_MAP = {
    "ddr": download_ddr,
    "deepdrid": download_deepdrid,
    "idrid": download_idrid,
    "idrid_segmentation": download_idrid_segmentation,
    "messidor2": download_messidor2,
}


def download_dataset(dataset_key: str, target_dir: Optional[Path] = None) -> Path:
    """Dispatches dataset download to the appropriate cohort downloader function."""
    key = dataset_key.strip().lower()
    if key in DOWNLOADER_MAP:
        return DOWNLOADER_MAP[key](target_dir=target_dir)
    if key in ("aptos", "eyepacs"):
        raise ValueError(
            f"Dataset '{dataset_key}' requires Kaggle competition rules agreement or manual acquisition. "
            f"Please download it via Kaggle CLI or competition webpage into data/raw/{key}."
        )
    raise ValueError(f"Unknown dataset '{dataset_key}'. Supported for auto-download: {list(DOWNLOADER_MAP.keys())}")


__all__ = [
    "download_ddr",
    "download_deepdrid",
    "download_idrid",
    "download_idrid_segmentation",
    "download_messidor2",
    "download_dataset",
]
