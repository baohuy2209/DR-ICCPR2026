"""Dataset preparation and consolidation for DeepDRiD."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.patient_ids import parse_patient_and_side
from drdg.paths import PROCESSED_DIR, RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_deepdrid(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[MasterFundusRecord]:
    """Consolidates DeepDRiD training, validation, and challenge subsets."""
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "deepdrid"
    out_dir = output_dir if output_dir is not None else PROCESSED_DIR / "deepdrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sub = src_dir / "DeepDRiD-master/regular_fundus_images"
    if not base_sub.exists():
        base_sub = src_dir

    train_csv = base_sub / "regular-fundus-training/regular-fundus-training.csv"
    val_csv = base_sub / "regular-fundus-validation/regular-fundus-validation.csv"

    records: List[MasterFundusRecord] = []
    dfs_to_process = []
    if train_csv.exists():
        dfs_to_process.append((pd.read_csv(train_csv), base_sub / "regular-fundus-training/Images"))
    if val_csv.exists():
        dfs_to_process.append((pd.read_csv(val_csv), base_sub / "regular-fundus-validation/Images"))

    for df, img_base in dfs_to_process:
        label_col = "patient_DR_Level" if "patient_DR_Level" in df.columns else "DR_Levels"
        if label_col not in df.columns:
            continue

        for _, row in df.iterrows():
            img_id = str(row["image_id"])
            diag_val = row[label_col]
            if pd.isna(diag_val):
                continue
            diag = int(diag_val)
            if diag not in (0, 1, 2, 3, 4):
                continue

            patient_id, side = parse_patient_and_side(img_id, "deepdrid")
            raw_patient = img_id.split("_")[0]
            raw_path = img_base / raw_patient / f"{img_id}.jpg"
            if not raw_path.exists():
                raw_path = img_base / f"{img_id}.jpg"
            if not raw_path.exists():
                continue

            native_path = out_dir / f"deepdrid_{img_id}.jpg"

            records.append(
                MasterFundusRecord(
                    image_id=f"deepdrid_{img_id}",
                    diagnosis=diag,
                    dataset_name="DeepDRiD",
                    dataset_key="deepdrid",
                    raw_full_path=str(raw_path.resolve()),
                    native_path=str(native_path.resolve()),
                    patient_id=patient_id,
                    side=side,
                    original_image_id=img_id,
                )
            )

    logger.info(f"Prepared {len(records)} records for DeepDRiD.")
    return records
