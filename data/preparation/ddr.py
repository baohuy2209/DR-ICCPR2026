"""Dataset preparation and consolidation for DDR."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.patient_ids import parse_patient_and_side
from drdg.paths import PROCESSED_DIR, RAW_DIR
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_ddr(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[MasterFundusRecord]:
    """Scans and validates DDR fundus images and ground-truth labels."""
    src_dir = raw_dir if raw_dir is not None else RAW_DIR / "ddr"
    out_dir = output_dir if output_dir is not None else PROCESSED_DIR / "ddr"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[MasterFundusRecord] = []

    # Look for annotation files (train.txt, valid.txt, test.txt or csv)
    label_files = list(src_dir.glob("**/*grading*.txt")) + list(src_dir.glob("**/*.csv")) + list(src_dir.glob("*.txt"))
    if not label_files:
        logger.warning(f"DDR annotations not found in {src_dir}. Skipping preparation.")
        return []

    for l_file in label_files:
        if l_file.suffix == ".txt":
            try:
                df = pd.read_csv(l_file, sep=r"\s+", header=None, names=["image_name", "diagnosis"])
            except Exception:
                continue
        else:
            try:
                df = pd.read_csv(l_file)
                # Normalize column names
                col_map = {c: c.lower() for c in df.columns}
                df.rename(columns=col_map, inplace=True)
                if "image_name" not in df.columns and "image_id" in df.columns:
                    df["image_name"] = df["image_id"]
                if "diagnosis" not in df.columns and "dr_grade" in df.columns:
                    df["diagnosis"] = df["dr_grade"]
            except Exception:
                continue

        if "image_name" not in df.columns or "diagnosis" not in df.columns:
            continue

        for _, row in df.iterrows():
            img_name = str(row["image_name"]).strip()
            try:
                diag = int(row["diagnosis"])
            except (ValueError, TypeError):
                continue

            if diag not in (0, 1, 2, 3, 4):
                continue

            base_name = Path(img_name).stem
            patient_id, side = parse_patient_and_side(base_name, "ddr")

            # Look for image file in raw_dir
            img_matches = list(src_dir.glob(f"**/{base_name}.*"))
            raw_path = img_matches[0] if img_matches else (src_dir / f"{base_name}.jpg")

            native_path = out_dir / f"ddr_{base_name}.jpg"

            records.append(
                MasterFundusRecord(
                    image_id=f"ddr_{base_name}",
                    diagnosis=diag,
                    dataset_name="DDR",
                    dataset_key="ddr",
                    raw_full_path=str(raw_path),
                    native_path=str(native_path),
                    patient_id=patient_id,
                    side=side,
                    original_image_id=base_name,
                )
            )

    logger.info(f"Prepared {len(records)} DDR records.")
    return records
