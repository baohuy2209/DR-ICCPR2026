"""Sliding-window patch extraction and foreground anchor caching for local MIL."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from drdg.data.fov import compute_fov_mask


def compute_valid_anchors(
    img_shape: Tuple[int, int],
    fov_mask: np.ndarray,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
) -> np.ndarray:
    """Calculates 2D top-left coordinate anchors [y0, x0] meeting the FOV threshold.

    Args:
        img_shape: (H, W) or (H, W, C) of source image.
        fov_mask: Boolean/binary mask of shape (H, W).
        patch_size: Square patch side length (p).
        stride: Step size between consecutive sliding window steps.
        fov_thresh: Minimum fraction of pixels within the FOV mask (default: 0.5).

    Returns:
        (N, 2) numpy array of integer top-left anchor coordinates [y0, x0].
    """
    h, w = img_shape[:2]
    p = patch_size

    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_coords(curr_stride: int) -> List[List[int]]:
        s = max(1, curr_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]

        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        valid_list = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = min(y0 + p, h), min(x0 + p, w)
                mask_patch = fov_mask[y0:y1, x0:x1]
                fov_ratio = float(mask_patch.mean()) if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list

    anchors = get_coords(stride)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.array(anchors, dtype=np.int32)


def extract_mil_patches_sliding(
    img_np: np.ndarray,
    fov_mask: Optional[np.ndarray] = None,
    patch_size: int = 128,
    stride: int = 96,
    fov_thresh: float = 0.5,
    mode: str = "train",
    num_sample: Union[int, Tuple[int, int]] = 48,
    max_patches_eval: int = 256,
    valid_anchors: Optional[np.ndarray] = None,
    return_anchors: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
    """Extracts sliding-window patches from native-resolution fundus images.

    Args:
        img_np: Input image array of shape (H, W, 3).
        fov_mask: Optional binary mask of shape (H, W).
        patch_size: Square patch size (default 128).
        stride: Sliding stride (default 96).
        fov_thresh: Minimum foreground ratio.
        mode: 'train' (random subsample) or 'eval' (deterministic uniform sampling).
        num_sample: Fixed int or (min, max) tuple of patches to sample in training.
        max_patches_eval: Upper limit on evaluated patches in evaluation mode.
        valid_anchors: Pre-cached (N, 2) array of coordinates.
        return_anchors: If True, also returns the coordinates of selected patches.

    Returns:
        tensor_patches: Tensor of shape (K, 3, patch_size, patch_size) in uint8 or float32.
        (optional) selected_coords: (K, 2) numpy array of anchor coordinates.
    """
    h, w, _ = img_np.shape
    p = patch_size

    if mode == "train":
        if isinstance(num_sample, (tuple, list)):
            m_target = int(np.random.randint(num_sample[0], num_sample[1] + 1))
        else:
            m_target = int(num_sample)
    else:
        m_target = max_patches_eval

    # Fast path if pre-computed valid_anchors are provided
    if valid_anchors is not None and len(valid_anchors) > 0:
        n_anchors = len(valid_anchors)
        if mode == "train":
            if n_anchors < m_target:
                idx = np.random.choice(n_anchors, size=m_target, replace=True)
            else:
                idx = np.random.choice(n_anchors, size=m_target, replace=False)
            selected_coords = valid_anchors[idx]
        else:
            max_eval = min(n_anchors, max_patches_eval)
            idx = np.linspace(0, n_anchors - 1, max_eval).astype(int)
            selected_coords = valid_anchors[idx]

        patches = []
        for y0, x0 in selected_coords:
            y1, x1 = min(y0 + p, h), min(x0 + p, w)
            patch = img_np[y0:y1, x0:x1]
            if patch.shape[0] < p or patch.shape[1] < p:
                pad_h = max(0, p - patch.shape[0])
                pad_w = max(0, p - patch.shape[1])
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            patches.append(patch)

        patches_np = np.stack(patches, axis=0)
        tensor_patches = torch.from_numpy(patches_np).permute(0, 3, 1, 2).contiguous()
        if return_anchors:
            return tensor_patches, np.asarray(selected_coords, dtype=np.int32)
        return tensor_patches

    # On-the-fly calculation if no anchors are provided
    if fov_mask is None:
        fov_mask = compute_fov_mask(img_np)

    anchors = compute_valid_anchors(
        img_shape=(h, w),
        fov_mask=fov_mask,
        patch_size=patch_size,
        stride=stride,
        fov_thresh=fov_thresh,
    )

    return extract_mil_patches_sliding(
        img_np=img_np,
        fov_mask=fov_mask,
        patch_size=patch_size,
        stride=stride,
        fov_thresh=fov_thresh,
        mode=mode,
        num_sample=num_sample,
        max_patches_eval=max_patches_eval,
        valid_anchors=anchors,
        return_anchors=return_anchors,
    )


def compute_foreground_ratio(patch_mask: np.ndarray) -> float:
    """Computes the foreground fraction of a patch mask."""
    if patch_mask.size == 0:
        return 0.0
    return float(np.mean(patch_mask > 0))


def filter_foreground_patches(
    patches: np.ndarray,
    fov_mask: np.ndarray,
    anchors: np.ndarray,
    patch_size: int = 128,
    min_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters patches according to their foreground ratio within the FOV mask."""
    valid_indices = []
    for i, (y0, x0) in enumerate(anchors):
        patch_m = fov_mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
        if compute_foreground_ratio(patch_m) >= min_ratio:
            valid_indices.append(i)
    if not valid_indices:
        return patches, anchors
    idx = np.array(valid_indices)
    return patches[idx], anchors[idx]


class ForegroundPatchCache:
    """In-memory or on-disk cache for computed patch anchor coordinates."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}

    def get(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieves cached anchors for an image ID."""
        if image_id in self._memory_cache:
            return self._memory_cache[image_id]
        if self.cache_dir:
            p = self.cache_dir / f"{image_id}_anchors.npy"
            if p.exists():
                arr = np.load(p)
                self._memory_cache[image_id] = arr
                return arr
        return None

    def set(self, image_id: str, anchors: np.ndarray) -> None:
        """Stores anchors for an image ID."""
        self._memory_cache[image_id] = anchors
        if self.cache_dir:
            p = self.cache_dir / f"{image_id}_anchors.npy"
            np.save(p, anchors)

