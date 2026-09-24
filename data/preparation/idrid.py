"""Dataset preparation and consolidation for IDRiD."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.patient_ids import parse_patient_and_side
from drdg.paths import PROCESSED_DIR, RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_idrid(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[MasterFundusRecord]:
    """Consolidates IDRiD grading images and ground truths."""
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "idrid"
    out_dir = output_dir if output_dir is not None else PROCESSED_DIR / "idrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_grading = src_dir / "Disease Grading"
    if not base_grading.exists():
        base_grading = src_dir

    train_csv = base_grading / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = base_grading / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"

    subsets = []
    if train_csv.exists():
        subsets.append((pd.read_csv(train_csv), base_grading / "1. Original Images/a. Training Set"))
    if test_csv.exists():
        subsets.append((pd.read_csv(test_csv), base_grading / "1. Original Images/b. Testing Set"))

    records: List[MasterFundusRecord] = []
    for df, img_base in subsets:
        id_col = next((c for c in df.columns if "image" in c.lower() or "name" in c.lower()), "Image name")
        label_col = next((c for c in df.columns if "retinopathy" in c.lower() or "grade" in c.lower()), "Retinopathy grade")

        for _, row in df.iterrows():
            img_id = str(row[id_col]).strip()
            diag_val = row[label_col]
            if pd.isna(diag_val):
                continue
            diag = int(diag_val)
            if diag not in (0, 1, 2, 3, 4):
                continue

            raw_path = None
            for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                candidate = img_base / f"{img_id}{ext}"
                if candidate.exists():
                    raw_path = candidate
                    break

            if raw_path is None:
                continue

            patient_id, side = parse_patient_and_side(img_id, "idrid")
            native_path = out_dir / f"idrid_{img_id}.jpg"

            records.append(
                MasterFundusRecord(
                    image_id=f"idrid_{img_id}",
                    diagnosis=diag,
                    dataset_name="IDRiD",
                    dataset_key="idrid",
                    raw_full_path=str(raw_path.resolve()),
                    native_path=str(native_path.resolve()),
                    patient_id=patient_id,
                    side=side,
                    original_image_id=img_id,
                )
            )

    logger.info(f"Prepared {len(records)} records for IDRiD.")
    return records
