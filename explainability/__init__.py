"""Explainable AI (XAI) modules: Grad-CAM, MIL Attention Saliency, IDRiD evaluation, and metrics."""

from drdg.explainability.gradcam import GradCAM
from drdg.explainability.idrid_dataset import IDRiDLesionDataset
from drdg.explainability.metrics import (
    area_matched_saliency_recall,
    evaluate_xai_sample,
    pixel_auroc,
    pointing_game,
)
from drdg.explainability.mil_saliency import MILAttentionSaliency

__all__ = [
    "GradCAM",
    "MILAttentionSaliency",
    "IDRiDLesionDataset",
    "pointing_game",
    "area_matched_saliency_recall",
    "pixel_auroc",
    "evaluate_xai_sample",
]
