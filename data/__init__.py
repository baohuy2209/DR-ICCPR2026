"""Data pipeline, cohort registries, preprocessing, transforms, datasets, and LODO splitting."""

from drdg.data.balancing import balance_training_records, compute_class_weights
from drdg.data.collate import trainer_compatible_collate_fn
from drdg.data.datasets import FundusDualBranchDataset
from drdg.data.fov import (
    compute_morphological_fov_mask,
    compute_soft_fov_mask,
    extract_circular_fov_region,
    filter_features_by_fov,
)
from drdg.data.lodo import LODOFold, generate_and_save_lodo_splits, generate_lodo_folds
from drdg.data.metadata import MasterFundusRecord, records_to_dataframe, validate_records
from drdg.data.patches import (
    ForegroundPatchCache,
    compute_foreground_ratio,
    compute_valid_anchors,
    extract_mil_patches_sliding,
    filter_foreground_patches,
)
from drdg.data.patient_ids import (
    generate_namespaced_patient_id,
    normalize_cohort_patient_id,
    parse_patient_id,
)
from drdg.data.preprocessing import (
    apply_ben_graham_enhancement,
    apply_clahe,
    apply_green_channel_ablation,
    crop_fundus_roi,
    extract_green_channel,
)
from drdg.data.registry import COHORT_REGISTRY, CohortSpec, get_cohort_spec
from drdg.data.splits import PatientSplit, audit_patient_leakage, split_patient_stratified
from drdg.data.transforms import (
    Denormalize,
    DualBranchTransform,
    get_eval_transforms,
    get_global_augmentations,
    get_patch_augmentations,
)

__all__ = [
    "COHORT_REGISTRY",
    "CohortSpec",
    "get_cohort_spec",
    "MasterFundusRecord",
    "validate_records",
    "records_to_dataframe",
    "parse_patient_id",
    "generate_namespaced_patient_id",
    "normalize_cohort_patient_id",
    "crop_fundus_roi",
    "extract_green_channel",
    "apply_ben_graham_enhancement",
    "apply_clahe",
    "apply_green_channel_ablation",
    "compute_morphological_fov_mask",
    "compute_soft_fov_mask",
    "extract_circular_fov_region",
    "filter_features_by_fov",
    "compute_valid_anchors",
    "compute_foreground_ratio",
    "filter_foreground_patches",
    "extract_mil_patches_sliding",
    "ForegroundPatchCache",
    "get_global_augmentations",
    "get_patch_augmentations",
    "DualBranchTransform",
    "Denormalize",
    "get_eval_transforms",
    "FundusDualBranchDataset",
    "trainer_compatible_collate_fn",
    "balance_training_records",
    "compute_class_weights",
    "PatientSplit",
    "audit_patient_leakage",
    "split_patient_stratified",
    "LODOFold",
    "generate_lodo_folds",
    "generate_and_save_lodo_splits",
]
