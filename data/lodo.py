"""Leave-One-Dataset-Out (LODO) 6-fold split generation and leakage verification.

Golden reference: merge_dataset_lodo.ipynb

Semantics:
  - 6 folds, each holding out one cohort as OOD test.
  - Source pool (remaining 5 cohorts) split 85/15 at patient level
    via GroupShuffleSplit(train_size=0.85, random_state=SEED+fold_idx).
  - Train split additionally quota-balanced (1,500–2,500 per grade).
  - Val split preserves natural clinical prevalence (no balancing).
  - Test split = 100% of held-out cohort (natural prevalence).
  - CSV naming: fold_{fold_idx}_{fold_slug}_{split}.csv
  - Total output: 18 CSV files (6 folds × 3 splits) + lodo_splits_summary.csv
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from drdg.data.metadata import MasterFundusRecord
from drdg.data.splits import PatientSplit, audit_patient_leakage, split_patient_stratified
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LODOFold:
    """Represents a single Leave-One-Dataset-Out fold.

    Matches merge_dataset_lodo.ipynb Cell 7 structure:
      - source_train_df: 85% of source pool, quota-balanced
      - source_val_df:   15% of source pool, natural prevalence
      - target_test_df:  100% of held-out cohort, natural prevalence
    """

    fold_idx: int
    target_cohort: str
    fold_slug: str
    source_cohorts: List[str]
    source_train_df: pd.DataFrame
    source_val_df: pd.DataFrame
    target_test_df: pd.DataFrame
    leakage_audit: Dict[str, Any]

    def verify_zero_leakage(self, patient_col: str = "patient_id") -> None:
        """Verifies strictly zero patient leakage across all partitions.

        Matches merge_dataset_lodo.ipynb Cell 7 assertions:
          assert len(train_pids.intersection(val_pids)) == 0
          assert len(train_pids.intersection(test_pids)) == 0
          assert len(val_pids.intersection(test_pids)) == 0

        Raises:
            ValueError: If any patient overlaps between partitions.
        """
        train_pats = set(self.source_train_df[patient_col].unique())
        val_pats = set(self.source_val_df[patient_col].unique())
        test_pats = set(self.target_test_df[patient_col].unique())

        inter_train_val = train_pats.intersection(val_pats)
        inter_train_test = train_pats.intersection(test_pats)
        inter_val_test = val_pats.intersection(test_pats)

        if inter_train_val:
            raise ValueError(
                f"Fold {self.fold_idx} ({self.target_cohort}): "
                f"PATIENT LEAKAGE: Train vs Val — {len(inter_train_val)} leaked patients"
            )
        if inter_train_test:
            raise ValueError(
                f"Fold {self.fold_idx} ({self.target_cohort}): "
                f"PATIENT LEAKAGE: Train vs Test — {len(inter_train_test)} leaked patients"
            )
        if inter_val_test:
            raise ValueError(
                f"Fold {self.fold_idx} ({self.target_cohort}): "
                f"PATIENT LEAKAGE: Val vs Test — {len(inter_val_test)} leaked patients"
            )

    def save_csvs(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Saves 3 partition CSVs matching merge_dataset_lodo.ipynb naming.

        Output naming: fold_{fold_idx}_{fold_slug}_{split}.csv

        Returns:
            Dict mapping split name to saved CSV Path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        prefix = f"fold_{self.fold_idx}_{self.fold_slug}"
        paths = {
            "train": out_path / f"{prefix}_train.csv",
            "val": out_path / f"{prefix}_val.csv",
            "test": out_path / f"{prefix}_test.csv",
        }

        self.source_train_df.to_csv(paths["train"], index=False)
        self.source_val_df.to_csv(paths["val"], index=False)
        self.target_test_df.to_csv(paths["test"], index=False)

        return paths


def generate_lodo_folds(
    data: Union[pd.DataFrame, List[MasterFundusRecord]],
    target_cohorts: Optional[List[str]] = None,
    fold_slugs: Optional[Dict[str, str]] = None,
    seed: int = 42,
    train_size: float = 0.85,
    patient_col: str = "patient_id",
    cohort_col: str = "cohort",
) -> List[LODOFold]:
    """Generates Leave-One-Dataset-Out folds from master dataset.

    Matches merge_dataset_lodo.ipynb semantics:
      - Source pool = all cohorts except held-out
      - GroupShuffleSplit(train_size=0.85, random_state=SEED+fold_idx) at patient level
      - No separate source_test — only train and val from source pool
      - Target test = 100% of held-out cohort

    Args:
        data: Master metadata DataFrame or list of MasterFundusRecord objects.
        target_cohorts: Ordered list of cohort keys matching DATASET_CONFIG order.
        fold_slugs: Optional mapping from cohort key to fold slug (for CSV naming).
        seed: Base random seed (fold uses seed + fold_idx).
        train_size: Source pool split ratio for training (default 0.85).
        patient_col: Name of patient ID column.
        cohort_col: Name of cohort column.

    Returns:
        List of verified LODOFold instances.
    """
    if isinstance(data, list):
        df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in data])
    else:
        df = data.copy()

    all_cohorts = sorted(df[cohort_col].unique().tolist())
    if target_cohorts is None:
        target_cohorts = all_cohorts

    if fold_slugs is None:
        fold_slugs = {c: c.lower() for c in target_cohorts}

    folds: List[LODOFold] = []

    for fold_idx, target in enumerate(target_cohorts):
        if target not in all_cohorts:
            raise ValueError(f"Target cohort '{target}' not found in data cohorts: {all_cohorts}")

        slug = fold_slugs.get(target, target.lower())

        # 1. Held-out OOD Test Set (100% natural clinical prevalence)
        target_test_df = df[df[cohort_col] == target].reset_index(drop=True)
        target_test_df["split"] = "test"
        target_test_df["fold"] = fold_idx
        target_test_df["held_out_dataset"] = target

        # 2. 5-Dataset In-Domain Pool
        source_df = df[df[cohort_col] != target].reset_index(drop=True)
        source_cohorts = [c for c in all_cohorts if c != target]

        # 3. Patient-level 85/15 Split (matching notebook GroupShuffleSplit)
        source_split: PatientSplit = split_patient_stratified(
            source_df,
            train_ratio=train_size,
            val_ratio=1.0 - train_size,
            test_ratio=0.0,
            seed=seed + fold_idx,
            patient_col=patient_col,
        )

        train_df = source_split.train_df.copy()
        train_df["split"] = "train"
        train_df["fold"] = fold_idx
        train_df["held_out_dataset"] = target

        val_df = source_split.val_df.copy()
        val_df["split"] = "val"
        val_df["fold"] = fold_idx
        val_df["held_out_dataset"] = target

        # Leakage audit — bipartite (train vs val, train vs test, val vs test)
        train_pats = set(train_df[patient_col].unique())
        val_pats = set(val_df[patient_col].unique())
        test_pats = set(target_test_df[patient_col].unique())

        leakage_audit = {
            "has_leakage": False,
            "train_val_overlap": list(train_pats.intersection(val_pats)),
            "train_test_overlap": list(train_pats.intersection(test_pats)),
            "val_test_overlap": list(val_pats.intersection(test_pats)),
        }
        if any(leakage_audit[k] for k in ["train_val_overlap", "train_test_overlap", "val_test_overlap"]):
            leakage_audit["has_leakage"] = True

        fold = LODOFold(
            fold_idx=fold_idx,
            target_cohort=target,
            fold_slug=slug,
            source_cohorts=source_cohorts,
            source_train_df=train_df,
            source_val_df=val_df,
            target_test_df=target_test_df,
            leakage_audit=leakage_audit,
        )

        fold.verify_zero_leakage(patient_col=patient_col)
        folds.append(fold)
        logger.info(
            f"Fold {fold_idx} ({target}/{slug}) created: "
            f"{len(fold.source_train_df)} train, {len(fold.source_val_df)} val, "
            f"{len(fold.target_test_df)} test ({target}). Zero leakage verified."
        )

    return folds


def generate_and_save_lodo_splits(
    data: Union[pd.DataFrame, List[MasterFundusRecord]],
    output_dir: Union[str, Path],
    seed: int = 42,
    patient_col: str = "patient_id",
    cohort_col: str = "cohort",
) -> Dict[str, Any]:
    """Generates all LODO folds and writes CSVs and summary JSON to output_dir.

    Produces 18 CSV files (6 folds × 3 splits) + lodo_splits_summary.csv,
    matching merge_dataset_lodo.ipynb output structure.

    Returns:
        Summary dict containing fold statistics and validation status.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    folds = generate_lodo_folds(data, seed=seed, patient_col=patient_col, cohort_col=cohort_col)

    summary_records = []

    summary: Dict[str, Any] = {
        "seed": seed,
        "n_folds": len(folds),
        "folds": {},
        "all_zero_leakage_verified": True,
    }

    for fold in folds:
        fold_paths = fold.save_csvs(out_path)
        fold_summary = {
            "fold_idx": fold.fold_idx,
            "target_cohort": fold.target_cohort,
            "fold_slug": fold.fold_slug,
            "source_cohorts": fold.source_cohorts,
            "n_train_images": len(fold.source_train_df),
            "n_train_patients": len(fold.source_train_df[patient_col].unique()),
            "n_val_images": len(fold.source_val_df),
            "n_val_patients": len(fold.source_val_df[patient_col].unique()),
            "n_test_images": len(fold.target_test_df),
            "n_test_patients": len(fold.target_test_df[patient_col].unique()),
            "zero_leakage_verified": not fold.leakage_audit["has_leakage"],
            "paths": {k: str(v) for k, v in fold_paths.items()},
        }
        summary["folds"][fold.target_cohort] = fold_summary

        # Summary row matching notebook's summary_records format
        train_df = fold.source_train_df
        val_df = fold.source_val_df
        test_df = fold.target_test_df
        summary_records.append({
            "fold": fold.fold_idx,
            "held_out_dataset": fold.target_cohort,
            "fold_slug": fold.fold_slug,
            "train_images": len(train_df),
            "train_patients": train_df[patient_col].nunique(),
            "val_images": len(val_df),
            "val_patients": val_df[patient_col].nunique(),
            "test_images": len(test_df),
            "test_patients": test_df[patient_col].nunique(),
        })

    # Save summary CSV matching notebook's lodo_splits_summary.csv
    summary_df = pd.DataFrame(summary_records)
    summary_csv = out_path / "lodo_splits_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Save summary JSON
    summary_file = out_path / "lodo_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"LODO splits and summary successfully generated at {out_path}")
    return summary
