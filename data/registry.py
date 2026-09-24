"""Cohort registry defining metadata, paths, and clinical protocol specifications."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CohortSpec:
    """Clinical and organizational specifications for a fundus image cohort."""

    dataset_key: str
    display_name: str
    country: str
    camera_device: str
    field_of_view: str
    grading_protocol: str
    id_col: str
    label_col: str
    id_suffix: str = ""
    csv_relative_path: str = ""
    raw_img_dir: str = ""


COHORT_REGISTRY: Dict[str, CohortSpec] = {
    "aptos": CohortSpec(
        dataset_key="aptos",
        display_name="APTOS 2019",
        country="India",
        camera_device="Zeiss, Topcon",
        field_of_view="45°",
        grading_protocol="Single-grader ICDRS",
        id_col="id_code",
        label_col="diagnosis",
        id_suffix=".png",
        csv_relative_path="data/raw/aptos/train.csv",
        raw_img_dir="data/raw/aptos/train_images",
    ),
    "ddr": CohortSpec(
        dataset_key="ddr",
        display_name="DDR",
        country="China",
        camera_device="Canon CR-2, Topcon",
        field_of_view="45°",
        grading_protocol="Multi-expert consensus",
        id_col="image_id",
        label_col="dr_level",
        id_suffix=".jpg",
        csv_relative_path="data/raw/ddr/train.csv",
        raw_img_dir="data/raw/ddr/train",
    ),
    "deepdrid": CohortSpec(
        dataset_key="deepdrid",
        display_name="DeepDRiD",
        country="China",
        camera_device="Topcon TRC-NW400",
        field_of_view="45°",
        grading_protocol="2-field paired stereoscopic",
        id_col="image_id",
        label_col="patient_DR_Level",
        id_suffix=".jpg",
        csv_relative_path="data/raw/deepdrid/metadata.csv",
        raw_img_dir="data/raw/deepdrid/images",
    ),
    "idrid": CohortSpec(
        dataset_key="idrid",
        display_name="IDRiD",
        country="India",
        camera_device="Kowa VX-10alpha",
        field_of_view="50°",
        grading_protocol="Dual-retina specialist panel",
        id_col="Image name",
        label_col="Retinopathy grade",
        id_suffix=".jpg",
        csv_relative_path="data/raw/idrid/train.csv",
        raw_img_dir="data/raw/idrid/images",
    ),
    "messidor2": CohortSpec(
        dataset_key="messidor2",
        display_name="Messidor-2",
        country="France",
        camera_device="Topcon TRC NW6",
        field_of_view="45°",
        grading_protocol="Adjudicated multi-grader panel",
        id_col="image_id",
        label_col="adjudicated_dr_grade",
        id_suffix=".jpg",
        csv_relative_path="data/raw/messidor2/messidor_data.csv",
        raw_img_dir="data/raw/messidor2/images",
    ),
    "eyepacs": CohortSpec(
        dataset_key="eyepacs",
        display_name="EyePACS",
        country="USA",
        camera_device="Centervue, Canon, Topcon",
        field_of_view="45°",
        grading_protocol="Tele-ophthalmology reading network",
        id_col="image",
        label_col="level",
        id_suffix=".jpeg",
        csv_relative_path="data/raw/eyepacs/trainLabels.csv",
        raw_img_dir="data/raw/eyepacs/train",
    ),
}


def get_cohort_spec(dataset_key: str) -> CohortSpec:
    """Retrieves CohortSpec for a valid dataset key."""
    key = dataset_key.strip().lower()
    if key not in COHORT_REGISTRY:
        raise KeyError(
            f"Unknown dataset_key '{dataset_key}'. Registered keys: {list(COHORT_REGISTRY.keys())}"
        )
    return COHORT_REGISTRY[key]


def list_registered_cohorts() -> List[str]:
    """Returns list of registered dataset keys."""
    return list(COHORT_REGISTRY.keys())
