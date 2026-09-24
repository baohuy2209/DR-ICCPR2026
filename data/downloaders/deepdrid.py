"""Downloader for DeepDRiD dataset."""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.base import download_kaggle_dataset


def download_deepdrid(target_dir: Optional[Path] = None) -> Path:
    """Downloads DeepDRiD dataset from Kaggle."""
    return download_kaggle_dataset(
        dataset_handle="yoctoman/deepdrid",
        target_dir=target_dir,
        dataset_name="deepdrid",
    )
