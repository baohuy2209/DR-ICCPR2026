"""Training-only class quota balancing and inverse frequency weighting."""

from typing import List, Union

import numpy as np
import pandas as pd
import torch

from drdg.data.metadata import MasterFundusRecord


def balance_training_records(
    data: Union[pd.DataFrame, List[MasterFundusRecord]],
    target_per_class: int = 2000,
    num_classes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Balances class distributions strictly for the training split.

    Validation and testing splits must preserve natural clinical prevalence
    and must NEVER be passed through this balancing routine.

    Args:
        data: DataFrame or list of MasterFundusRecord instances.
        target_per_class: Target sample count for each of the 5 severity grades.
        num_classes: Number of disease stages (default 5).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Balanced pandas DataFrame containing target_per_class records per grade.
    """
    if isinstance(data, list):
        df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in data])
    else:
        df = data.copy()

    balanced_dfs = []
    rng = np.random.default_rng(seed)

    for grade in range(num_classes):
        subset = df[df["diagnosis"] == grade].copy()
        current_count = len(subset)
        if current_count == 0:
            continue

        if current_count >= target_per_class:
            sampled_idx = rng.choice(subset.index, size=target_per_class, replace=False)
        else:
            sampled_idx = rng.choice(subset.index, size=target_per_class, replace=True)

        balanced_dfs.append(df.loc[sampled_idx])

    if not balanced_dfs:
        return df

    balanced = pd.concat(balanced_dfs, ignore_index=True)
    # Deterministic shuffle
    shuffled_idx = rng.permutation(len(balanced))
    return balanced.iloc[shuffled_idx].reset_index(drop=True)


def compute_class_weights(
    data: Union[pd.DataFrame, List[MasterFundusRecord]],
    num_classes: int = 5,
) -> torch.Tensor:
    """Computes normalized inverse-frequency class weights for loss function rebalancing."""
    if isinstance(data, list):
        labels = [r.diagnosis if hasattr(r, "diagnosis") else r["diagnosis"] for r in data]
    else:
        labels = data["diagnosis"].tolist()

    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = total / (num_classes * max(1, counts[c]))

    # Normalize weights so mean is 1.0
    weights = weights / (weights.mean() + 1e-7)
    return torch.from_numpy(weights).float()
