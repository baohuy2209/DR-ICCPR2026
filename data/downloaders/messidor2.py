"""Downloader for Messidor-2 dataset."""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.base import download_kaggle_dataset


def download_messidor2(target_dir: Optional[Path] = None) -> Path:
    """Downloads Messidor-2 dataset from Kaggle."""
    return download_kaggle_dataset(
        dataset_handle="mariaherrerot/messidor2preprocess",
        target_dir=target_dir,
        dataset_name="messidor2",
    )
