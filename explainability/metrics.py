"""Quantitative explainability evaluation metrics: Pointing Game, Area-Matched Saliency Recall, and pixel AUROC."""

from typing import Any, Dict, List, Union

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import auc, roc_curve


def pointing_game(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
    tolerance_pixels: int = 15,
) -> float:
    """Evaluates whether the peak saliency point falls within or near a ground-truth lesion.

    Args:
        saliency_map: Continuous 2D heatmap of shape (H, W).
        lesion_mask: Binary boolean or {0, 1} mask of shape (H, W).
        tolerance_pixels: Euclidean distance tolerance around ground truth lesions.

    Returns:
        1.0 for a hit (maximum within lesion + tolerance), 0.0 for a miss.
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=bool)

    if not np.any(l_mask):
        return float("nan")

    # Locate the peak saliency point
    y_peak, x_peak = np.unravel_index(np.argmax(s_map), s_map.shape)

    if tolerance_pixels <= 0:
        return 1.0 if l_mask[y_peak, x_peak] else 0.0

    # Expand lesion mask using Euclidean distance transform or dilation
    dist = distance_transform_edt(~l_mask)
    return 1.0 if dist[y_peak, x_peak] <= tolerance_pixels else 0.0


def area_matched_saliency_recall(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
) -> float:
    """Computes the fraction of lesion area covered by the top-k saliency pixels.

    The top fraction k is set exactly equal to the ground-truth lesion area proportion,
    eliminating bias from arbitrary binarization thresholds.

    Args:
        saliency_map: 2D heatmap of shape (H, W).
        lesion_mask: 2D binary ground-truth mask.

    Returns:
        Recall fraction in [0.0, 1.0].
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=bool)

    lesion_area = int(np.sum(l_mask))
    total_pixels = l_mask.size

    if lesion_area == 0 or total_pixels == 0:
        return float("nan")

    # Find threshold for top-k pixels where k = lesion_area
    flat_s = s_map.ravel()
    k_th_value = np.partition(flat_s, -lesion_area)[-lesion_area]
    top_k_mask = s_map >= k_th_value

    overlap = np.sum(top_k_mask & l_mask)
    return float(overlap / lesion_area)


def pixel_auroc(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
) -> float:
    """Computes pixel-wise Area Under the Receiver Operating Characteristic curve.

    Args:
        saliency_map: Continuous 2D heatmap of shape (H, W).
        lesion_mask: Binary boolean mask of shape (H, W).

    Returns:
        Pixel AUROC score in [0.0, 1.0].
    """
    s_map = np.asarray(saliency_map, dtype=float)
    l_mask = np.asarray(lesion_mask, dtype=int)

    if len(np.unique(l_mask)) < 2:
        return float("nan")

    # Subsample if resolution is very large to avoid memory bottlenecks
    if s_map.size > 262144:  # > 512x512
        step = int(np.ceil(np.sqrt(s_map.size / 262144)))
        s_map = s_map[::step, ::step]
        l_mask = l_mask[::step, ::step]

    fpr, tpr, _ = roc_curve(l_mask.ravel(), s_map.ravel())
    score = float(auc(fpr, tpr))
    return score if not np.isnan(score) else 0.5


def evaluate_xai_sample(
    saliency_map: np.ndarray,
    lesion_mask: np.ndarray,
    tolerance_pixels: int = 15,
) -> Dict[str, float]:
    """Computes all three quantitative XAI metrics for a single image-saliency pair."""
    return {
        "pointing_game_accuracy": pointing_game(saliency_map, lesion_mask, tolerance_pixels=tolerance_pixels),
        "area_matched_saliency_recall": area_matched_saliency_recall(saliency_map, lesion_mask),
        "pixel_auroc": pixel_auroc(saliency_map, lesion_mask),
    }
