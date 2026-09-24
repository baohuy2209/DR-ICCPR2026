"""Bipartite evaluation serialization schema and validation for LODO cross-domain benchmarks.

Golden reference: main_model_lodo.ipynb Cell 17 Section 7.

The notebook records two evaluation partitions per fold:
  - in_domain_val: In-domain source validation metrics + calibrated RDR threshold
  - ood_test: OOD held-out cohort metrics evaluated with frozen threshold

There is NO separate "source_test" partition in the original research design.
The prior tripartite schema (source_validation / source_test / target_test) was
an invention that did not match the actual notebook implementation.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class MetricBundle:
    """Standardized metric evaluation bundle for a single partition."""

    qwk: float
    accuracy: float
    within_1_accuracy: float
    per_grade_sensitivity: Dict[str, float]
    rdr_auc: float
    rdr_sensitivity_at_95_specificity: float
    total_samples: int
    confusion_matrix: Optional[List[List[int]]] = None

    def __post_init__(self) -> None:
        """Validates field constraints."""
        # Ensure per_grade_sensitivity has keys '0', '1', '2', '3', '4'
        for g in ["0", "1", "2", "3", "4"]:
            if g not in self.per_grade_sensitivity:
                # Try integer key fallback if passed
                if int(g) in self.per_grade_sensitivity:
                    self.per_grade_sensitivity[g] = float(self.per_grade_sensitivity[int(g)])
                else:
                    self.per_grade_sensitivity[g] = 0.0
            else:
                self.per_grade_sensitivity[g] = float(self.per_grade_sensitivity[g])

        if self.total_samples < 0:
            raise ValueError(f"total_samples must be non-negative, got {self.total_samples}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts bundle to dictionary."""
        d = asdict(self)
        if self.confusion_matrix is None:
            d.pop("confusion_matrix", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricBundle":
        """Instantiates MetricBundle from dictionary."""
        return cls(
            qwk=float(data["qwk"]),
            accuracy=float(data["accuracy"]),
            within_1_accuracy=float(data["within_1_accuracy"]),
            per_grade_sensitivity={str(k): float(v) for k, v in data["per_grade_sensitivity"].items()},
            rdr_auc=float(data["rdr_auc"]),
            rdr_sensitivity_at_95_specificity=float(data["rdr_sensitivity_at_95_specificity"]),
            total_samples=int(data["total_samples"]),
            confusion_matrix=data.get("confusion_matrix"),
        )


@dataclass
class LODOFoldResult:
    """Bipartite result record matching main_model_lodo.ipynb Cell 17 semantics.

    Fields:
        in_domain_val: Metrics from source validation (5 cohorts, 15% holdout).
        ood_test: Metrics from OOD test (held-out cohort, 100% natural prevalence).
        threshold_rdr_spec95: Frozen RDR threshold calibrated on in_domain_val.
    """

    variant: str
    held_out_dataset: str
    fold: int
    seed: int
    threshold_rdr_spec95: float
    in_domain_val: MetricBundle
    ood_test: MetricBundle

    def __post_init__(self) -> None:
        """Validates LODO result schema constraints."""
        valid_variants = {"v0", "v1", "v2", "v3"}
        v = self.variant.strip().lower()
        if v not in valid_variants:
            raise ValueError(f"variant must be one of {valid_variants}, got: {self.variant}")
        self.variant = v

        if not (0 <= self.fold <= 5):
            raise ValueError(f"fold must be in [0, 5], got {self.fold}")

        if not (0.0 <= self.threshold_rdr_spec95 <= 1.0):
            raise ValueError(f"threshold_rdr_spec95 must be in [0.0, 1.0], got {self.threshold_rdr_spec95}")

        if not isinstance(self.in_domain_val, MetricBundle):
            if isinstance(self.in_domain_val, dict):
                self.in_domain_val = MetricBundle.from_dict(self.in_domain_val)
            else:
                raise TypeError("in_domain_val must be a MetricBundle instance or dict")

        if not isinstance(self.ood_test, MetricBundle):
            if isinstance(self.ood_test, dict):
                self.ood_test = MetricBundle.from_dict(self.ood_test)
            else:
                raise TypeError("ood_test must be a MetricBundle instance or dict")

    @property
    def delta_lodo(self) -> float:
        """Generalization drop: val_qwk - ood_qwk (matches notebook delta_lodo)."""
        return self.in_domain_val.qwk - self.ood_test.qwk

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to nested dictionary."""
        return {
            "variant": self.variant,
            "held_out_dataset": self.held_out_dataset,
            "fold": self.fold,
            "seed": self.seed,
            "threshold_rdr_spec95": self.threshold_rdr_spec95,
            "in_domain_val": self.in_domain_val.to_dict(),
            "ood_test": self.ood_test.to_dict(),
            "delta_lodo": self.delta_lodo,
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Serializes result to json file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_p

    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> "LODOFoldResult":
        """Loads and validates result from json file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            variant=str(data["variant"]),
            held_out_dataset=str(data["held_out_dataset"]),
            fold=int(data["fold"]),
            seed=int(data["seed"]),
            threshold_rdr_spec95=float(data["threshold_rdr_spec95"]),
            in_domain_val=MetricBundle.from_dict(data["in_domain_val"]),
            ood_test=MetricBundle.from_dict(data["ood_test"]),
        )


def validate_lodo_result_dict(data: Dict[str, Any]) -> bool:
    """Validates raw dictionary against LODOFoldResult schema."""
    LODOFoldResult(
        variant=str(data["variant"]),
        held_out_dataset=str(data["held_out_dataset"]),
        fold=int(data["fold"]),
        seed=int(data["seed"]),
        threshold_rdr_spec95=float(data["threshold_rdr_spec95"]),
        in_domain_val=MetricBundle.from_dict(data["in_domain_val"]),
        ood_test=MetricBundle.from_dict(data["ood_test"]),
    )
    return True
