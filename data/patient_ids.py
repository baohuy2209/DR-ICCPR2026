"""Patient identifier parsing and cohort namespacing."""

import re
from typing import Tuple


def parse_patient_and_side(image_id: str, dataset_key: str) -> Tuple[str, str]:
    """Extracts namespaced patient ID and lateral eye side from image identifier.

    Rules across cohorts:
        - aptos: Unique patient per image. Namespaced as 'aptos::<image_id_base>'.
        - ddr: Hyphenated pattern '001-0001-000.jpg' -> patient is '001-0001'.
        - deepdrid: Format '001_l1.jpg' -> patient is '001', side is 'left'.
        - idrid: Format 'IDRiD_001.jpg' -> patient is '001'.
        - messidor2: Format '20051019_38557_0100_PP.tif' -> patient is '20051019'.
        - eyepacs: Format '10_left.jpeg' -> patient is '10', side is 'left'.

    Args:
        image_id: Raw or cleaned filename / image identifier string.
        dataset_key: Cohort identifier ('aptos', 'ddr', 'deepdrid', 'idrid', 'messidor2', 'eyepacs').

    Returns:
        namespaced_patient_id: Formatted as '<dataset_key>::<patient_id>'.
        side: One of 'left', 'right', 'unknown'.
    """
    img_str = str(image_id).strip().lower()
    k = dataset_key.strip().lower()

    # Determine lateral side first
    if "left" in img_str or "_l" in img_str:
        side = "left"
    elif "right" in img_str or "_r" in img_str:
        side = "right"
    else:
        side = "unknown"

    base = re.sub(r"\.(jpg|jpeg|png|tif|tiff)$", "", img_str, flags=re.IGNORECASE)

    if "idrid" in k:
        parts = base.split("_")
        pid = parts[-1] if len(parts) > 1 else base
    elif "eyepacs" in k or "deepdrid" in k:
        pid = base.split("_")[0]
    elif "ddr" in k:
        parts = base.split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else base
    elif "messidor" in k:
        pid = base.split("_")[0]
    else:
        pid = base

    return f"{dataset_key}::{pid}", side


def parse_patient_id(image_id: str, dataset_key: str) -> str:
    """Extracts only the namespaced patient ID."""
    pid, _ = parse_patient_and_side(image_id, dataset_key)
    return pid


def generate_namespaced_patient_id(cohort: str, local_patient_id: str) -> str:
    """Formats a namespaced patient ID with the double colon separator."""
    return f"{cohort.strip().lower()}::{str(local_patient_id).strip()}"


def normalize_cohort_patient_id(patient_id: str, cohort: str) -> str:
    """Ensures patient_id is properly namespaced."""
    if "::" in patient_id:
        return patient_id
    return generate_namespaced_patient_id(cohort, patient_id)

