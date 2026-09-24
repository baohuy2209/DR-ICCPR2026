"""Base helper for automated Kaggle dataset downloading."""

import os
import shutil
from pathlib import Path
from typing import Optional

import kagglehub

from drdg.paths import RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def download_kaggle_dataset(
    dataset_handle: str,
    target_dir: Optional[Path] = None,
    dataset_name: str = "dataset",
) -> Path:
    """Downloads a dataset from Kaggle via kagglehub and moves it to the target directory.

    Args:
        dataset_handle: Kaggle dataset handle (e.g. 'mariaherrerot/ddrdataset').
        target_dir: Optional destination path. If None, defaults to RAW_DIR / dataset_name.
        dataset_name: Short name for logging.

    Returns:
        Path to the local directory containing the dataset files.
    """
    dest = target_dir if target_dir is not None else RAW_DIR / dataset_name
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_name} ({dataset_handle}) via kagglehub...")
    cache_path = kagglehub.dataset_download(dataset_handle)
    logger.info(f"Cache downloaded to: {cache_path}. Syncing to: {dest}...")

    for item in Path(cache_path).iterdir():
        target_item = dest / item.name
        if item.is_dir():
            if target_item.exists():
                shutil.rmtree(target_item)
            shutil.copytree(item, target_item)
        else:
            shutil.copy2(item, target_item)

    logger.info(f"Successfully staged {dataset_name} at: {dest.resolve()}")
    return dest
