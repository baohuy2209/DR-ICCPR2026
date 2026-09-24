"""Evaluation metrics for diabetic retinopathy grading."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_curve, auc


def calculate_referable_sensitivity_at_specificity(
    y_true: np.ndarray,
    referable_probs: np.ndarray,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict[str, float]:
    """Computes clinical Sensitivity at fixed Specificity operating points for Referable DR screening.

    Referable threshold: Grade >= 2 (Moderate, Severe, PDR vs No DR, Mild).
    """
    y_true_ref = (np.asarray(y_true, dtype=int) >= 2).astype(int)
    referable_probs = np.asarray(referable_probs, dtype=float)

    results: Dict[str, float] = {}
    n_pos = int(y_true_ref.sum())
    n_neg = int((1 - y_true_ref).sum())

    if n_pos == 0 or n_neg == 0:
        results["Referable_AUC"] = float("nan")
        for spec in target_specificities:
            pct = int(round(spec * 100))
            results[f"Ref_Sensitivity_at_Spec{pct}"] = float("nan")
            results[f"Ref_Threshold_at_Spec{pct}"] = float("nan")
        return results

    fpr, tpr, thresholds = roc_curve(y_true_ref, referable_probs)
    results["Referable_AUC"] = float(auc(fpr, tpr))
    specificity = 1.0 - fpr

    for spec in target_specificities:
        pct = int(round(spec * 100))
        valid_idx = np.where(specificity >= spec)[0]
        if len(valid_idx) == 0:
            sens_at_spec = 0.0
            best_thresh = float(thresholds[-1])
        else:
            best_sub_idx = valid_idx[np.argmax(tpr[valid_idx])]
            sens_at_spec = float(tpr[best_sub_idx])
            best_thresh = float(thresholds[best_sub_idx])
        results[f"Ref_Sensitivity_at_Spec{pct}"] = sens_at_spec
        results[f"Ref_Threshold_at_Spec{pct}"] = best_thresh

    return results


def calculate_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
    y_probs: Optional[np.ndarray] = None,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict[str, Any]:
    """Computes clinical and statistical evaluation metrics for 5-grade DR classification.

    Metrics:
        1. Quadratic Weighted Kappa (QWK)
        2. Exact Multi-class Accuracy
        3. Within-1-Grade Accuracy (|y_true - y_pred| <= 1)
        4. Confusion Matrix and per-grade recall (sensitivity)
        5. Referable DR (Grade >= 2) Sensitivity and Specificity
        6. Sensitivity at target specificities (80%, 85%, 90%, 95%) and Referable AUC
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if np.isnan(qwk):
        qwk = 0.0

    acc = float(accuracy_score(y_true, y_pred))
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_grade_recall: Dict[str, float] = {}
    for c in range(num_classes):
        denom = cm[c, :].sum()
        recall = float(cm[c, c] / denom) if denom > 0 else 0.0
        per_grade_recall[f"Recall_Grade_{c}"] = recall

    y_true_ref = (y_true >= 2).astype(int)
    y_pred_ref = (y_pred >= 2).astype(int)

    cm_ref = confusion_matrix(y_true_ref, y_pred_ref, labels=[0, 1])
    if cm_ref.size == 4:
        tn, fp, fn, tp = cm_ref.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    ref_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    result: Dict[str, Any] = {
        "QWK": qwk,
        "Accuracy": acc,
        "Within_1_Accuracy": within_1,
        "Within_1_Grade_Acc": within_1,
        "Referable_Sensitivity": ref_sensitivity,
        "Referable_Specificity": ref_specificity,
        "Confusion_Matrix": cm.tolist(),
        **per_grade_recall,
    }

    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float)
        if y_probs.ndim == 2 and y_probs.shape[1] >= 3:
            referable_probs = y_probs[:, 2:].sum(axis=1)
        else:
            referable_probs = y_probs.ravel()
        spec_metrics = calculate_referable_sensitivity_at_specificity(
            y_true, referable_probs, target_specificities=target_specificities
        )
        result.update(spec_metrics)

    return result


def extract_rank_probs(outputs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extracts raw cumulative rank probabilities P(y > k) from model outputs."""
    if "rank_probs" in outputs:
        rp = outputs["rank_probs"]
        return rp.detach().cpu().numpy() if hasattr(rp, "detach") else np.asarray(rp)
    if "cum_probs" in outputs:
        cp = outputs["cum_probs"]
        return cp.detach().cpu().numpy() if hasattr(cp, "detach") else np.asarray(cp)
    return None


def rank_monotonicity_violations(rank_probs: np.ndarray) -> float:
    """Computes violation rate where P(y > k) increases with k (should be non-increasing)."""
    if rank_probs.ndim != 2 or rank_probs.shape[1] <= 1:
        return 0.0
    diffs = np.diff(rank_probs, axis=1)  # Expected <= 0
    violations = np.any(diffs > 1e-4, axis=1)
    return float(np.mean(violations))
