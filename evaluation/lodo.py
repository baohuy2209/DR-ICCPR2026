"""Leave-One-Dataset-Out (LODO) evaluation engine with frozen threshold isolation.

Golden reference: main_model_lodo.ipynb Cell 17 Section 7.

The notebook performs BIPARTITE evaluation per fold:
  1. In-domain validation (source val) — compute QWK + calibrate RDR threshold tau*
  2. OOD test (held-out cohort) — compute QWK + evaluate sensitivity at frozen tau*
  3. Delta LODO = val_qwk - ood_qwk (generalization drop)

There is NO separate "source_test" partition in the research design.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drdg.evaluation.grade_metrics import calculate_per_grade_metrics
from drdg.evaluation.metrics import (
    calculate_dr_metrics,
    calculate_referable_sensitivity_at_specificity,
)
from drdg.evaluation.result_schema import LODOFoldResult, MetricBundle
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


def run_inference_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward evaluation on a DataLoader and extracts true labels, predictions, and class probabilities.

    Returns:
        y_true: (N,) ground truth integer labels.
        y_pred: (N,) predicted integer grades.
        y_probs: (N, 5) predicted normalized class probabilities.
    """
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            global_img = batch.get("global_img")
            if global_img is not None:
                global_img = global_img.to(device)

            patch_imgs = batch.get("patch_imgs")
            if patch_imgs is not None:
                patch_imgs = patch_imgs.to(device)

            patch_mask = batch.get("patch_mask")
            if patch_mask is not None:
                patch_mask = patch_mask.to(device)

            # Support diverse model signatures
            if hasattr(model, "forward"):
                if patch_imgs is not None and patch_mask is not None:
                    outputs = model(global_img=global_img, patch_imgs=patch_imgs, patch_mask=patch_mask)
                elif patch_imgs is not None:
                    outputs = model(global_img=global_img, patch_imgs=patch_imgs)
                elif global_img is not None:
                    outputs = model(global_img)
                else:
                    outputs = model(batch["image"].to(device))
            else:
                outputs = model(batch["image"].to(device))

            if isinstance(outputs, dict):
                logits = outputs.get("logits")
                probs = outputs.get("probs")
                preds = outputs.get("preds")
            else:
                logits = outputs
                probs = None
                preds = None

            if probs is None:
                probs = torch.softmax(logits, dim=-1)

            if preds is None:
                preds = torch.argmax(probs, dim=-1)

            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=int)
    y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 5), dtype=float)

    return y_true, y_pred, y_probs


def evaluate_partition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    frozen_threshold_spec95: Optional[float] = None,
) -> Tuple[MetricBundle, float]:
    """Computes comprehensive MetricBundle for a partition.

    Args:
        y_true: (N,) ground truth grades.
        y_pred: (N,) predicted discrete grades.
        y_probs: (N, 5) predicted normalized class probabilities.
        frozen_threshold_spec95: If provided, evaluates sensitivity using this fixed threshold.
                                 If None, calibrates threshold on this partition.

    Returns:
        MetricBundle instance and the calibrated or used threshold.
    """
    total_samples = len(y_true)
    if total_samples == 0:
        empty_sens = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
        return MetricBundle(0.0, 0.0, 0.0, empty_sens, 0.0, 0.0, 0), 0.5

    # General multi-class metrics
    base_metrics = calculate_dr_metrics(y_true, y_pred, y_probs=y_probs)
    grade_info = calculate_per_grade_metrics(y_true, y_pred)

    per_grade_sens = {
        str(c): float(grade_info["per_grade"][str(c)]["sensitivity"]) for c in range(5)
    }

    # Referable DR probability: sum of Grade 2, 3, 4
    referable_probs = np.sum(y_probs[:, 2:], axis=1) if y_probs.shape[1] >= 5 else y_probs[:, -1]
    y_true_ref = (y_true >= 2).astype(int)

    ref_stats = calculate_referable_sensitivity_at_specificity(
        y_true, referable_probs, target_specificities=(0.95,)
    )
    rdr_auc = float(ref_stats.get("Referable_AUC", 0.0))
    if np.isnan(rdr_auc):
        rdr_auc = 0.0

    calibrated_thresh = float(ref_stats.get("Ref_Threshold_at_Spec95", 0.5))
    if np.isnan(calibrated_thresh):
        calibrated_thresh = 0.5

    # Determine threshold to evaluate
    eval_thresh = frozen_threshold_spec95 if frozen_threshold_spec95 is not None else calibrated_thresh

    # Compute sensitivity at eval_thresh
    pos_mask = y_true_ref == 1
    if np.sum(pos_mask) > 0:
        sens_at_95 = float(np.mean(referable_probs[pos_mask] >= eval_thresh))
    else:
        sens_at_95 = 0.0

    bundle = MetricBundle(
        qwk=float(base_metrics.get("QWK", 0.0)),
        accuracy=float(base_metrics.get("Accuracy", 0.0)),
        within_1_accuracy=float(base_metrics.get("Within_1_Accuracy", base_metrics.get("Within_1_Grade_Acc", 0.0))),
        per_grade_sensitivity=per_grade_sens,
        rdr_auc=rdr_auc,
        rdr_sensitivity_at_95_specificity=sens_at_95,
        total_samples=total_samples,
        confusion_matrix=grade_info["confusion_matrix"],
    )

    return bundle, eval_thresh


def evaluate_lodo_fold(
    model: nn.Module,
    source_val_loader: DataLoader,
    ood_test_loader: DataLoader,
    device: torch.device,
    variant: str,
    held_out_dataset: str,
    fold: int,
    seed: int = 42,
) -> LODOFoldResult:
    """Orchestrates bipartite evaluation for a single LODO fold.

    Matches main_model_lodo.ipynb Cell 17 Section 7:
      Step 1: Evaluate in-domain validation data → compute val QWK + calibrate
              frozen RDR decision threshold tau* at >= 95% specificity.
      Step 2: Evaluate OOD held-out test data using frozen tau* → compute OOD QWK.
      Delta LODO = val_qwk - ood_qwk.
    """
    logger.info(f"Evaluating LODO Fold {fold} (Held-out: {held_out_dataset}, Variant: {variant})...")

    # 1. In-domain validation — calibrate threshold
    val_true, val_pred, val_probs = run_inference_on_loader(model, source_val_loader, device)
    val_bundle, frozen_thresh = evaluate_partition(val_true, val_pred, val_probs, frozen_threshold_spec95=None)
    logger.info(f"In-Domain Val QWK: {val_bundle.qwk:.4f} | Calibrated RDR threshold at 95% spec: {frozen_thresh:.4f}")

    # 2. OOD Test (zero-shot domain transfer) — evaluate with frozen threshold
    ood_true, ood_pred, ood_probs = run_inference_on_loader(model, ood_test_loader, device)
    ood_bundle, _ = evaluate_partition(ood_true, ood_pred, ood_probs, frozen_threshold_spec95=frozen_thresh)

    delta_lodo = val_bundle.qwk - ood_bundle.qwk
    logger.info(
        f"OOD Test ({held_out_dataset}) QWK: {ood_bundle.qwk:.4f} | "
        f"RDR Sens@95: {ood_bundle.rdr_sensitivity_at_95_specificity:.4f} | "
        f"Delta LODO: {delta_lodo:.4f}"
    )

    fold_result = LODOFoldResult(
        variant=variant,
        held_out_dataset=held_out_dataset,
        fold=fold,
        seed=seed,
        threshold_rdr_spec95=frozen_thresh,
        in_domain_val=val_bundle,
        ood_test=ood_bundle,
    )

    return fold_result
