"""Patient-level stratified splitting and zero-leakage auditing."""

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from drdg.data.metadata import MasterFundusRecord
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PatientSplit:
    """Disjoint patient-level partition holding train, val, and test subsets."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_patients: Set[str]
    val_patients: Set[str]
    test_patients: Set[str]

    def verify_zero_leakage(self) -> None:
        """Asserts mathematically that patient partitions have zero intersection."""
        inter_train_val = self.train_patients.intersection(self.val_patients)
        inter_train_test = self.train_patients.intersection(self.test_patients)
        inter_val_test = self.val_patients.intersection(self.test_patients)

        if inter_train_val:
            raise ValueError(f"Patient leakage detected between train and val: {inter_train_val}")
        if inter_train_test:
            raise ValueError(f"Patient leakage detected between train and test: {inter_train_test}")
        if inter_val_test:
            raise ValueError(f"Patient leakage detected between val and test: {inter_val_test}")


def audit_patient_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_col: str = "patient_id",
) -> Dict[str, Any]:
    """Audits train, validation, and test partitions for patient ID leakage.

    Returns:
        Dict with leakage statistics, counts, and boolean has_leakage flag.
    """
    train_pats = set(train_df[patient_col].unique())
    val_pats = set(val_df[patient_col].unique())
    test_pats = set(test_df[patient_col].unique())

    inter_train_val = train_pats.intersection(val_pats)
    inter_train_test = train_pats.intersection(test_pats)
    inter_val_test = val_pats.intersection(test_pats)

    has_leakage = bool(inter_train_val or inter_train_test or inter_val_test)

    return {
        "has_leakage": has_leakage,
        "n_train_patients": len(train_pats),
        "n_val_patients": len(val_pats),
        "n_test_patients": len(test_pats),
        "leakage_train_val": list(inter_train_val),
        "leakage_train_test": list(inter_train_test),
        "leakage_val_test": list(inter_val_test),
    }


def split_patient_stratified(
    data: Union[pd.DataFrame, List[MasterFundusRecord]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    patient_col: str = "patient_id",
) -> PatientSplit:
    """Partitions dataset into disjoint train/val/test splits strictly at patient level.

    All fundus photographs from the same patient (e.g. left eye, right eye, multiple visits)
    are strictly isolated to a single partition to eliminate patient-level data leakage.

    Args:
        data: Master metadata DataFrame or list of MasterFundusRecord instances.
        train_ratio: Fraction for training split (default 0.70).
        val_ratio: Fraction for validation split (default 0.15).
        test_ratio: Fraction for evaluation test split (default 0.15).
        seed: Random seed for deterministic reproducibility.
        patient_col: Column name containing namespaced patient IDs.

    Returns:
        PatientSplit dataclass holding verified disjoint splits and sets of patient IDs.
    """
    if isinstance(data, list):
        df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in data])
    else:
        df = data.copy()

    # Step 1: Split Train (70%) vs Temp (30%)
    temp_ratio = val_ratio + test_ratio
    gss_train = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_train.split(df, groups=df[patient_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Step 2: Split Temp into Val and Test
    if test_ratio <= 0.0:
        val_df = temp_df
        test_df = pd.DataFrame(columns=df.columns)
    else:
        val_relative_ratio = val_ratio / temp_ratio
        gss_test = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_relative_ratio), random_state=seed)
        val_sub_idx, test_sub_idx = next(gss_test.split(temp_df, groups=temp_df[patient_col]))
        val_df = temp_df.iloc[val_sub_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_sub_idx].reset_index(drop=True)

    split = PatientSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_patients=set(train_df[patient_col].unique()),
        val_patients=set(val_df[patient_col].unique()),
        test_patients=set(test_df[patient_col].unique()),
    )

    split.verify_zero_leakage()
    logger.info(
        f"Patient split generated: {len(train_df)} train ({len(split.train_patients)} patients), "
        f"{len(val_df)} val ({len(split.val_patients)} patients), "
        f"{len(test_df)} test ({len(split.test_patients)} patients). Zero leakage verified."
    )
    return split
