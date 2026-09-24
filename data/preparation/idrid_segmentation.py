"""Dataset preparation for IDRiD pixel-level lesion segmentation masks."""

from pathlib import Path
from typing import Dict, List, Optional

from drdg.paths import RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)

LESION_TYPES = ["MA", "HE", "EX", "SE"]


def prepare_idrid_segmentation(
    raw_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Path]]:
    """Maps IDRiD segmentation image IDs to their respective lesion mask filepaths.

    Lesion categories:
        - MA: Microaneurysms
        - HE: Haemorrhages
        - EX: Hard Exudates
        - SE: Soft Exudates

    Returns:
        Dict mapping image_id (e.g. 'IDRiD_01') to dict of lesion_type -> mask_path.
    """
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "idrid_segmentation"
    mask_base = src_dir / "All Segmentation Groundtruths/a. Training Set"
    if not mask_base.exists():
        mask_base = src_dir

    mask_map: Dict[str, Dict[str, Path]] = {}

    for lesion in LESION_TYPES:
        lesion_dir = mask_base / lesion
        if not lesion_dir.exists():
            # Check subfolder pattern e.g. "1. Microaneurysms"
            for folder in mask_base.glob(f"*{lesion}*"):
                if folder.is_dir():
                    lesion_dir = folder
                    break

        if not lesion_dir.exists():
            continue

        for mask_file in lesion_dir.glob("*.tif*"):
            img_id = mask_file.stem.split("_")[0] + "_" + mask_file.stem.split("_")[1]
            if img_id not in mask_map:
                mask_map[img_id] = {}
            mask_map[img_id][lesion] = mask_file

    logger.info(f"Discovered {len(mask_map)} images with lesion segmentation masks in IDRiD.")
    return mask_map
