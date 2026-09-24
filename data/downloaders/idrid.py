"""Downloader for IDRiD grading dataset."""

from pathlib import Path
from typing import Optional

from drdg.data.downloaders.base import download_kaggle_dataset


def download_idrid(target_dir: Optional[Path] = None) -> Path:
    """Downloads IDRiD dataset from Kaggle."""
    return download_kaggle_dataset(
        dataset_handle="abdullahshafi315/indian-diabetic-retinopathy-image-datasetidrid",
        target_dir=target_dir,
        dataset_name="idrid",
    )
