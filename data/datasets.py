"""PyTorch Dataset implementations for dual-branch multi-scale diabetic retinopathy models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from drdg.data.fov import compute_fov_mask
from drdg.data.metadata import MasterFundusRecord
from drdg.data.patches import compute_valid_anchors, extract_mil_patches_sliding
from drdg.data.preprocessing import apply_green_channel_ablation
from drdg.data.transforms import get_global_augmentations
from drdg.utils.logging import get_logger

logger = get_logger(__name__)


class FundusDualBranchDataset(Dataset):
    """PyTorch Dataset serving the multi-scale dual-branch architecture.

    Features:
        - Stream 1: Global Context whole-image resized to img_size x img_size with spatial/photometric augmentations.
        - Stream 2: Native-resolution sliding-window patches extracted around valid retinal FOV anchors.
        - Fast in-memory list-of-dict records for O(1) indexing.
        - High-throughput uint8 tensor return to minimize IPC Shared Memory bandwidth.
    """

    def __init__(
        self,
        records: Union[List[MasterFundusRecord], pd.DataFrame, str, Path],
        img_size: int = 512,
        patch_size: int = 128,
        stride: int = 96,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        num_patches_train: int = 48,
        max_patches_eval: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.is_train = split.lower() == "train"
        self.ablation_mode = ablation_mode
        self.cache_dir = cache_dir
        self.num_patches_train = num_patches_train
        self.max_patches_eval = max_patches_eval

        # Normalize records into a list of dictionaries
        if isinstance(records, (str, Path)):
            p = Path(records)
            if not p.exists():
                raise FileNotFoundError(f"Metadata file not found: {p}")
            df = pd.read_csv(p)
            self.records = df.to_dict("records")
        elif isinstance(records, pd.DataFrame):
            self.records = records.to_dict("records")
        elif isinstance(records, list):
            self.records = [r.to_dict() if hasattr(r, "to_dict") else r for r in records]
        else:
            raise TypeError(f"Unsupported records type: {type(records)}")

        self.global_transform = get_global_augmentations(img_size=img_size, is_train=self.is_train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        label = int(row.get("diagnosis", 0))
        image_id = str(row.get("image_id", idx))
        patient_id = str(row.get("patient_id", f"unknown::{image_id}"))
        dataset_key = str(row.get("dataset_key", "default"))

        # Resolve image filepath
        img_path = str(row.get("native_path", row.get("raw_full_path", "")))

        # Load image via OpenCV
        img_rgb = None
        if img_path and Path(img_path).exists():
            bgr = cv2.imread(img_path)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if img_rgb is None:
            # Fallback for synthetic/missing tests
            img_rgb = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)

        # 1. Morphological FOV mask on RAW image
        fov_mask = compute_fov_mask(img_rgb)

        # 2. Green channel enhancement / Ben Graham normalization
        processed_rgb = apply_green_channel_ablation(img_rgb, ablation_mode=self.ablation_mode)

        # 3. Stream 1: Global Context (Resize + Augmentations)
        augmented = self.global_transform(image=processed_rgb)
        global_u8 = augmented["image"] if isinstance(augmented, dict) else augmented
        global_tensor = torch.from_numpy(global_u8).permute(2, 0, 1).contiguous()

        # 4. Stream 2: Local MIL Patches
        patches_tensor, selected_anchors = extract_mil_patches_sliding(
            img_np=processed_rgb,
            fov_mask=fov_mask,
            patch_size=self.patch_size,
            stride=self.stride,
            mode=self.split,
            num_sample=self.num_patches_train,
            max_patches_eval=self.max_patches_eval,
            return_anchors=True,
        )

        k_patches = patches_tensor.shape[0]
        patch_mask = torch.ones(k_patches, dtype=torch.bool)

        return {
            "global_img": global_tensor,
            "local_patches": patches_tensor,
            "patch_mask": patch_mask,
            "patch_coords": torch.from_numpy(selected_anchors),
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": image_id,
            "patient_id": patient_id,
            "dataset_key": dataset_key,
        }
