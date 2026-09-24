"""Canonical metadata record schema and validation for harmonized fundus datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from drdg.data.registry import COHORT_REGISTRY


@dataclass
class MasterFundusRecord:
    """Standardized record for an individual fundus photograph across all six cohorts."""

    image_id: str
    diagnosis: int
    dataset_name: str
    dataset_key: str
    raw_full_path: str
    native_path: str
    patient_id: str
    side: str = "unknown"
    original_image_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates all field constraints according to data model specifications."""
        if not self.image_id or not isinstance(self.image_id, str):
            raise ValueError(f"image_id must be a non-empty string, got: {self.image_id}")

        if not isinstance(self.diagnosis, int) or self.diagnosis not in (0, 1, 2, 3, 4):
            raise ValueError(f"diagnosis must be integer in {{0, 1, 2, 3, 4}}, got: {self.diagnosis}")

        key = self.dataset_key.strip().lower()
        if key not in COHORT_REGISTRY:
            raise ValueError(f"dataset_key '{self.dataset_key}' not in registered cohorts: {list(COHORT_REGISTRY.keys())}")
        self.dataset_key = key

        if "::" not in self.patient_id:
            raise ValueError(f"patient_id must be namespaced with '::' (e.g. 'aptos::001'), got: {self.patient_id}")

        valid_sides = {"left", "right", "unknown"}
        self.side = self.side.strip().lower()
        if self.side not in valid_sides:
            raise ValueError(f"side must be one of {valid_sides}, got: {self.side}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MasterFundusRecord":
        """Constructs record from dictionary."""
        return cls(
            image_id=str(data["image_id"]),
            diagnosis=int(data["diagnosis"]),
            dataset_name=str(data["dataset_name"]),
            dataset_key=str(data["dataset_key"]),
            raw_full_path=str(data["raw_full_path"]),
            native_path=str(data["native_path"]),
            patient_id=str(data["patient_id"]),
            side=str(data.get("side", "unknown")),
            original_image_id=str(data["original_image_id"]) if data.get("original_image_id") is not None else None,
        )


def records_to_dataframe(records: List[MasterFundusRecord]) -> pd.DataFrame:
    """Converts a list of MasterFundusRecord instances to a structured pandas DataFrame."""
    return pd.DataFrame([r.to_dict() for r in records])


def dataframe_to_records(df: pd.DataFrame) -> List[MasterFundusRecord]:
    """Converts a pandas DataFrame back into a list of validated MasterFundusRecord objects."""
    records = []
    for _, row in df.iterrows():
        records.append(MasterFundusRecord.from_dict(row.to_dict()))
    return records


def validate_records(records: Union[List[MasterFundusRecord], pd.DataFrame]) -> bool:
    """Validates that all records comply with the MasterFundusRecord schema.

    Returns:
        True if all records are valid.

    Raises:
        ValueError if any record is invalid.
    """
    if isinstance(records, pd.DataFrame):
        recs = dataframe_to_records(records)
    else:
        recs = records

    for rec in recs:
        if not isinstance(rec, MasterFundusRecord):
            raise ValueError(f"Expected MasterFundusRecord, got {type(rec)}")
    return True

