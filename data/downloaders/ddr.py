"""Downloader for DDR dataset."""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.base import download_kaggle_dataset


def download_ddr(target_dir: Optional[Path] = None) -> Path:
    """Downloads DDR dataset from Kaggle."""
    return download_kaggle_dataset(
        dataset_handle="mariaherrerot/ddrdataset",
        target_dir=target_dir,
        dataset_name="ddr",
    )
