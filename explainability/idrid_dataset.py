"""IDRiD pixel-level lesion segmentation dataset loader and composite mask builder."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from drdg.paths import RAW_DATA_DIR


class IDRiDLesionDataset(Dataset):
    """Dataset for IDRiD pixel-level diabetic retinopathy lesion annotations.

    Ground-truth lesion types:
    - Microaneurysms (MA)
    - Hemorrhages (HE)
    - Hard Exudates (EX)
    - Soft Exudates (SE)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = RAW_DATA_DIR / "idrid_segmentation",
        target_size: Tuple[int, int] = (512, 512),
        lesion_types: Tuple[str, ...] = ("MA", "HE", "EX", "SE"),
        use_synthetic_if_missing: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.lesion_types = lesion_types
        self.use_synthetic_if_missing = use_synthetic_if_missing

        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        if self.data_dir.exists():
            img_candidates = list(self.data_dir.glob("**/*Original_Images*/*.jpg")) + list(
                self.data_dir.glob("**/*.jpg")
            )
            for img_p in img_candidates:
                if "mask" in img_p.name.lower():
                    continue
                base_id = img_p.stem
                # Find matching lesion masks
                mask_paths = {}
                for l_type in self.lesion_types:
                    matches = list(self.data_dir.glob(f"**/*{l_type}*/*{base_id}*")) + list(
                        self.data_dir.glob(f"**/*{base_id}*{l_type}*")
                    )
                    if matches:
                        mask_paths[l_type] = matches[0]

                if mask_paths:
                    samples.append({
                        "image_id": base_id,
                        "image_path": img_p,
                        "mask_paths": mask_paths,
                    })

        if not samples and self.use_synthetic_if_missing:
            # Generate 10 synthetic samples for verification and dry runs
            for idx in range(10):
                samples.append({
                    "image_id": f"idrid_syn_{idx:03d}",
                    "image_path": None,
                    "mask_paths": {},
                    "synthetic": True,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]

        if info.get("synthetic", False) or info.get("image_path") is None:
            # Synthetic sample generator
            h, w = self.target_size
            img = np.zeros((h, w, 3), dtype=np.uint8)
            y, x = np.ogrid[:h, :w]
            disk = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= (min(h, w) // 2 - 20) ** 2
            img[disk] = [70, 150, 200]

            # Synthetic lesion mask
            mask = np.zeros((h, w), dtype=np.uint8)
            # Add a couple of synthetic lesion circles
            l1 = ((x - w // 2 - 40) ** 2 + (y - h // 2 - 30) ** 2) <= 15 ** 2
            l2 = ((x - w // 2 + 50) ** 2 + (y - h // 2 + 40) ** 2) <= 20 ** 2
            mask[l1 | l2] = 255

            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return {
                "image_id": info["image_id"],
                "image": tensor_img,
                "lesion_mask": mask > 0,
            }

        # Load real image
        bgr = cv2.imread(str(info["image_path"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]

        # Build union lesion mask
        composite_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for l_type, m_path in info["mask_paths"].items():
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                composite_mask = np.maximum(composite_mask, (m > 0).astype(np.uint8) * 255)

        # Resize to target size
        resized_img = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))
        resized_mask = cv2.resize(composite_mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        tensor_img = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0
        return {
            "image_id": info["image_id"],
            "image": tensor_img,
            "lesion_mask": resized_mask > 0,
        }
