"""Dataset preparation modules for raw cohort images and segmentation masks.

Exports:
- prepare_aptos
- prepare_ddr
- prepare_deepdrid
- prepare_idrid
- prepare_idrid_segmentation
- prepare_messidor2
- prepare_eyepacs
- prepare_dataset
- resize_image
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from drdg.data.metadata import MasterFundusRecord
from drdg.data.preparation.aptos import prepare_aptos
from drdg.data.preparation.ddr import prepare_ddr
from drdg.data.preparation.deepdrid import prepare_deepdrid
from drdg.data.preparation.eyepacs import prepare_eyepacs
from drdg.data.preparation.idrid import prepare_idrid
from drdg.data.preparation.idrid_segmentation import prepare_idrid_segmentation
from drdg.data.preparation.messidor2 import prepare_messidor2
from drdg.data.preparation.resize import resize_image

PREPARATION_MAP = {
    "aptos": prepare_aptos,
    "ddr": prepare_ddr,
    "deepdrid": prepare_deepdrid,
    "idrid": prepare_idrid,
    "idrid_segmentation": prepare_idrid_segmentation,
    "messidor2": prepare_messidor2,
    "eyepacs": prepare_eyepacs,
}


def prepare_dataset(
    dataset_key: str,
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[MasterFundusRecord]:
    """Prepares and validates fundus records for a specific cohort or all cohorts."""
    key = dataset_key.strip().lower().replace("-", "_")
    r_dir = Path(raw_dir) if raw_dir else None
    o_dir = Path(output_dir) if output_dir else None

    if key == "all":
        all_records: List[MasterFundusRecord] = []
        for cohort, fn in PREPARATION_MAP.items():
            if cohort == "idrid_segmentation":
                continue
            c_raw = (r_dir / cohort) if r_dir else None
            c_out = (o_dir / cohort) if o_dir else None
            records = fn(raw_dir=c_raw, output_dir=c_out)
            all_records.extend(records)
        return all_records

    if key not in PREPARATION_MAP:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Supported keys: {list(PREPARATION_MAP.keys())} or 'all'.")

    return PREPARATION_MAP[key](raw_dir=r_dir, output_dir=o_dir)


__all__ = [
    "prepare_aptos",
    "prepare_ddr",
    "prepare_deepdrid",
    "prepare_idrid",
    "prepare_idrid_segmentation",
    "prepare_messidor2",
    "prepare_eyepacs",
    "prepare_dataset",
    "resize_image",
    "PREPARATION_MAP",
]
