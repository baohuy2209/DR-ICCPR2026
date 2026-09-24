"""Dataset preparation and consolidation for EyePACS."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.patient_ids import parse_patient_and_side
from drdg.paths import PROCESSED_DIR, RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_eyepacs(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[MasterFundusRecord]:
    """Scans and validates EyePACS (Kaggle Diabetic Retinopathy Detection) images and labels."""
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "eyepacs"
    out_dir = output_dir if output_dir is not None else PROCESSED_DIR / "eyepacs"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = list(src_dir.glob("**/*labels*.csv")) + list(src_dir.glob("*.csv"))
    if not csv_candidates:
        logger.warning(f"EyePACS CSV not found in {src_dir}. Skipping preparation.")
        return []

    csv_path = csv_candidates[0]
    df = pd.read_csv(csv_path)

    col_map = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    id_col = next((c for c in ["image", "image_id", "id_code"] if c in df.columns), None)
    diag_col = next((c for c in ["level", "diagnosis", "dr_grade"] if c in df.columns), None)

    if not id_col or not diag_col:
        logger.warning(f"Required columns not found in EyePACS CSV: {df.columns}")
        return []

    records: List[MasterFundusRecord] = []
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        try:
            diag = int(row[diag_col])
        except (ValueError, TypeError):
            continue

        if diag not in (0, 1, 2, 3, 4):
            continue

        base_name = Path(raw_id).stem
        patient_id, side = parse_patient_and_side(base_name, "eyepacs")

        img_matches = list(src_dir.glob(f"**/{base_name}.*"))
        raw_path = img_matches[0] if img_matches else (src_dir / f"{base_name}.jpeg")
        native_path = out_dir / f"eyepacs_{base_name}.jpg"

        records.append(
            MasterFundusRecord(
                image_id=f"eyepacs_{base_name}",
                diagnosis=diag,
                dataset_name="EyePACS",
                dataset_key="eyepacs",
                raw_full_path=str(raw_path),
                native_path=str(native_path),
                patient_id=patient_id,
                side=side,
                original_image_id=base_name,
            )
        )

    logger.info(f"Prepared {len(records)} EyePACS records.")
    return records
