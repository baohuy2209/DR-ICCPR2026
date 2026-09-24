"""Ordinal severity metrics, distance-weighted accuracy, and consistency checks."""

from typing import Any, Dict, List, Optional

import numpy as np


def calculate_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cumulative_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes specialized metrics for ordinal severity grading.

    Args:
        y_true: Ground truth ordinal grades (N,).
        y_pred: Predicted ordinal grades (N,).
        cumulative_probs: Optional array of shape (N, K-1) representing cumulative thresholds P(Y > k).

    Returns:
        Dictionary of ordinal performance statistics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    mse = float(np.mean(errors ** 2))
    root_mse = float(np.sqrt(mse))

    # Error distribution: exact, off-by-1, off-by-2, off-by-3, off-by-4
    off_by = {}
    for d in range(5):
        off_by[f"off_by_{d}"] = float(np.mean(errors == d))

    severe_error_rate = float(np.mean(errors >= 2))

    results: Dict[str, Any] = {
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": root_mse,
        "severe_error_rate_ge_2": severe_error_rate,
        "error_distribution": off_by,
    }

    if cumulative_probs is not None:
        # Check monotonicity violations in predicted cumulative probabilities
        # Valid cumulative probs must satisfy: P(Y > 0) >= P(Y > 1) >= P(Y > 2) >= P(Y > 3)
        cp = np.asarray(cumulative_probs, dtype=float)
        if cp.ndim == 2 and cp.shape[1] > 1:
            diffs = cp[:, 1:] - cp[:, :-1]  # Should be <= 0 for non-increasing
            violations = np.sum(diffs > 1e-5, axis=1)
            violation_rate = float(np.mean(violations > 0))
            results["cumulative_monotonicity_violation_rate"] = violation_rate
        else:
            results["cumulative_monotonicity_violation_rate"] = 0.0

    return results
