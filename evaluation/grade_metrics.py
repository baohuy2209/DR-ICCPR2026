"""Detailed per-grade sensitivity, specificity, and adjacent error analysis."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_per_grade_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """Computes per-grade clinical diagnostics: Sensitivity, Specificity, Precision, F1.

    Args:
        y_true: Ground truth grade array of shape (N,).
        y_pred: Predicted discrete grade array of shape (N,).
        num_classes: Number of ordinal DR classes (default 5: 0 to 4).

    Returns:
        Dictionary with per-grade metrics and summary statistics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total_samples = int(len(y_true))

    per_grade = {}
    sensitivities = []
    specificities = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(total_samples - tp - fn - fp)

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

        per_grade[str(c)] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "support": int(tp + fn),
        }
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    within_1_acc = float(np.mean(np.abs(y_true - y_pred) <= 1)) if total_samples > 0 else 0.0
    within_2_acc = float(np.mean(np.abs(y_true - y_pred) <= 2)) if total_samples > 0 else 0.0
    exact_acc = float(np.mean(y_true == y_pred)) if total_samples > 0 else 0.0

    return {
        "per_grade": per_grade,
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "exact_accuracy": exact_acc,
        "within_1_accuracy": within_1_acc,
        "within_2_accuracy": within_2_acc,
        "confusion_matrix": cm.tolist(),
        "total_samples": total_samples,
    }
