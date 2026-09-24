"""Evaluation metrics, grade-specific statistics, and LODO tripartite orchestration."""

from drdg.evaluation.grade_metrics import calculate_per_grade_metrics
from drdg.evaluation.lodo import evaluate_lodo_fold, evaluate_partition, run_inference_on_loader
from drdg.evaluation.metrics import (
    calculate_dr_metrics,
    calculate_referable_sensitivity_at_specificity,
    extract_rank_probs,
    rank_monotonicity_violations,
)
from drdg.evaluation.ordinal_metrics import calculate_ordinal_metrics
from drdg.evaluation.result_schema import LODOFoldResult, MetricBundle, validate_lodo_result_dict
from drdg.evaluation.statistics import compute_metric_summary, paired_wilcoxon_test

__all__ = [
    "calculate_dr_metrics",
    "calculate_referable_sensitivity_at_specificity",
    "extract_rank_probs",
    "rank_monotonicity_violations",
    "calculate_per_grade_metrics",
    "calculate_ordinal_metrics",
    "paired_wilcoxon_test",
    "compute_metric_summary",
    "MetricBundle",
    "LODOFoldResult",
    "validate_lodo_result_dict",
    "evaluate_lodo_fold",
    "evaluate_partition",
    "run_inference_on_loader",
]
