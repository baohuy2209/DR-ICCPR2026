"""Dataset preparation and consolidation for APTOS 2019."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.patient_ids import parse_patient_and_side
from drdg.paths import PROCESSED_DIR, RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_aptos(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[MasterFundusRecord]:
    """Scans and validates APTOS 2019 fundus images and ground-truth labels."""
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "aptos"
    out_dir = output_dir if output_dir is not None else PROCESSED_DIR / "aptos"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = [
        src_dir / "train.csv",
        src_dir / "train_1.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        logger.warning(f"APTOS CSV not found in {src_dir}. Skipping preparation.")
        return []

    df = pd.read_csv(csv_path)
    img_dir_candidates = [
        src_dir / "train_images",
        src_dir / "train_images/train_images",
    ]
    img_dir = next((p for p in img_dir_candidates if p.exists()), src_dir / "train_images")

    records: List[MasterFundusRecord] = []
    for _, row in df.iterrows():
        img_id = str(row["id_code"])
        diag = int(row["diagnosis"])
        if diag not in (0, 1, 2, 3, 4):
            continue

        raw_path = img_dir / f"{img_id}.png"
        if not raw_path.exists():
            continue

        patient_id, side = parse_patient_and_side(img_id, "aptos")
        native_path = out_dir / f"aptos_{img_id}.jpg"

        records.append(
            MasterFundusRecord(
                image_id=f"aptos_{img_id}",
                diagnosis=diag,
                dataset_name="APTOS 2019",
                dataset_key="aptos",
                raw_full_path=str(raw_path.resolve()),
                native_path=str(native_path.resolve()),
                patient_id=patient_id,
                side=side,
                original_image_id=img_id,
            )
        )

    logger.info(f"Prepared {len(records)} records for APTOS 2019.")
    return records
