"""Downloader for IDRiD lesion segmentation dataset."""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.base import download_kaggle_dataset


def download_idrid_segmentation(target_dir: Optional[Path] = None) -> Path:
    """Downloads IDRiD lesion segmentation masks dataset from Kaggle."""
    return download_kaggle_dataset(
        dataset_handle="realhaadkhan/idrid-segmentation-dataset",
        target_dir=target_dir,
        dataset_name="idrid_segmentation",
    )
