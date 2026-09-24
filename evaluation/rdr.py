"""Referable Diabetic Retinopathy (RDR) screening calibration and evaluation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


@dataclass
class RDRCalibrationResult:
    """Stores calibrated threshold and source validation metrics."""

    target_specificity: float
    calibrated_threshold: float
    validation_sensitivity: float
    validation_specificity: float
    validation_auc: float
    total_val_samples: int


@dataclass
class RDREvaluationResult:
    """Stores evaluation metrics on a target/test set using a frozen threshold."""

    frozen_threshold: float
    test_sensitivity: float
    test_specificity: float
    test_auc: float
    total_test_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def calibrate_rdr_threshold(
    y_true_val: Union[List[int], np.ndarray],
    y_probs_val: Union[List[float], np.ndarray],
    target_specificity: float = 0.95,
) -> RDRCalibrationResult:
    """Calibrates screening operating threshold tau* strictly on source validation data.

    Referable DR is defined as Grade >= 2.
    Threshold is chosen such that validation specificity >= target_specificity (default: 0.95).

    Args:
        y_true_val: Validation ground truth discrete grades (0 to 4) or binary referable indicators.
        y_probs_val: Predicted referable probability (P(Y >= 2)) on validation set.
        target_specificity: Desired clinical specificity operating point (default: 0.95).

    Returns:
        RDRCalibrationResult holding frozen threshold and validation metrics.
    """
    y_true = np.asarray(y_true_val, dtype=int)
    # If grades 0-4 are passed, binarize to referable (>=2)
    if y_true.max() > 1:
        y_true_bin = (y_true >= 2).astype(int)
    else:
        y_true_bin = y_true

    probs = np.asarray(y_probs_val, dtype=float)
    n_samples = len(y_true_bin)

    if n_samples == 0 or len(np.unique(y_true_bin)) < 2:
        return RDRCalibrationResult(
            target_specificity=target_specificity,
            calibrated_threshold=0.5,
            validation_sensitivity=0.0,
            validation_specificity=1.0,
            validation_auc=float("nan"),
            total_val_samples=n_samples,
        )

    fpr, tpr, thresholds = roc_curve(y_true_bin, probs)
    val_auc = float(auc(fpr, tpr))
    specificity = 1.0 - fpr

    # Find operating points where specificity >= target_specificity
    valid_indices = np.where(specificity >= target_specificity)[0]

    if len(valid_indices) == 0:
        # Fallback to point of highest specificity
        best_idx = int(np.argmax(specificity))
    else:
        # Among points meeting target specificity, maximize sensitivity (TPR)
        best_idx = int(valid_indices[np.argmax(tpr[valid_indices])])

    threshold_star = float(thresholds[best_idx])
    # Thresholds in roc_curve can exceed 1.0 at index 0 (thresholds[0] = max(probs) + 1)
    if threshold_star > 1.0:
        threshold_star = 1.0
    elif threshold_star < 0.0:
        threshold_star = 0.0

    # Verify actual sensitivity and specificity achieved at threshold_star
    preds_at_thresh = (probs >= threshold_star).astype(int)
    cm = confusion_matrix(y_true_bin, preds_at_thresh, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    actual_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    actual_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return RDRCalibrationResult(
        target_specificity=target_specificity,
        calibrated_threshold=threshold_star,
        validation_sensitivity=actual_sens,
        validation_specificity=actual_spec,
        validation_auc=val_auc,
        total_val_samples=n_samples,
    )


def evaluate_rdr_with_frozen_threshold(
    y_true_test: Union[List[int], np.ndarray],
    y_probs_test: Union[List[float], np.ndarray],
    frozen_threshold: float,
) -> RDREvaluationResult:
    """Evaluates test data using a previously frozen operating threshold.

    Guarantees zero-leakage: the test data has no influence on the decision threshold.
    """
    y_true = np.asarray(y_true_test, dtype=int)
    n_samples = len(y_true)

    if n_samples == 0:
        return RDREvaluationResult(
            frozen_threshold=frozen_threshold,
            test_sensitivity=0.0,
            test_specificity=0.0,
            test_auc=float("nan"),
            total_test_samples=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
        )

    if y_true.max() > 1:
        y_true_bin = (y_true >= 2).astype(int)
    else:
        y_true_bin = y_true

    probs = np.asarray(y_probs_test, dtype=float)

    preds = (probs >= frozen_threshold).astype(int)
    cm = confusion_matrix(y_true_bin, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    test_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_bin)) >= 2:
        fpr, tpr, _ = roc_curve(y_true_bin, probs)
        test_auc = float(auc(fpr, tpr))
    else:
        test_auc = float("nan")

    return RDREvaluationResult(
        frozen_threshold=frozen_threshold,
        test_sensitivity=test_sens,
        test_specificity=test_spec,
        test_auc=test_auc,
        total_test_samples=n_samples,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
    )
