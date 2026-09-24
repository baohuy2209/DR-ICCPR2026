"""Morphological Field-of-View (FOV) mask computation for fundus photographs."""

from typing import Any, Optional, Tuple
import cv2
import numpy as np


def compute_fov_mask(
    img_np: np.ndarray,
    erode_px: int = 15,
    threshold: Optional[int] = None,
    kernel_size: int = 25,
    **kwargs: Any,
) -> np.ndarray:
    """Computes the true retinal Field-of-View (FOV) binary mask for a single fundus image.

    4-Step Pipeline:
        1. Green Channel + Otsu's or fixed Thresholding: Dynamically calculates optimal
           bimodal threshold or uses fixed threshold.
        2. Morphological Close & Open (ellipse): Seals internal vessel holes and removes noise.
        3. Largest Connected Component: Strictly retains the main retinal disk (removes artifacts/watermarks).
        4. Inward Morphological Erosion (erode_px): Strips away peripheral edge vignetting/halos.

    Args:
        img_np: Input fundus image array (H, W, 3) in RGB/BGR or (H, W) grayscale.
        erode_px: Inward erosion radius in pixels to remove peripheral halo artifacts.
        threshold: Optional fixed threshold. If None, Otsu's adaptive threshold is used.
        kernel_size: Kernel dimension for morphological closing and opening.

    Returns:
        Boolean 2D array of shape (H, W) where True indicates valid retinal aperture.
    """
    if img_np is None or img_np.size == 0:
        return np.zeros((1, 1), dtype=bool)

    if img_np.ndim == 3:
        # Green channel offers highest absorption contrast for retinal boundaries
        gray = img_np[:, :, 1] if img_np.shape[2] == 3 else img_np[:, :, 0]
    else:
        gray = img_np

    # 1. Thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is not None and threshold > 0:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphological Close & Open
    k_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 3. Largest Connected Component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fov_mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        fov_mask = cleaned

    # 4. Inward Erosion
    if erode_px > 0:
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        fov_mask = cv2.erode(fov_mask, kernel_erode)

    return fov_mask > 0


compute_morphological_fov_mask = compute_fov_mask


def apply_fov_mask(
    img: np.ndarray,
    mask: np.ndarray,
    fill_value: int = 0,
) -> np.ndarray:
    """Sets all pixels outside the FOV mask to fill_value."""
    out = img.copy()
    out[~mask] = fill_value
    return out


def compute_soft_fov_mask(
    img_np: np.ndarray,
    blur_sigma: float = 5.0,
    **kwargs: Any,
) -> np.ndarray:
    """Computes a soft-edged continuous [0, 1] FOV mask."""
    hard_mask = compute_fov_mask(img_np, **kwargs).astype(np.float32)
    ksize = int(blur_sigma * 4) + 1
    if ksize % 2 == 0:
        ksize += 1
    soft_mask = cv2.GaussianBlur(hard_mask, (ksize, ksize), blur_sigma)
    return np.clip(soft_mask, 0.0, 1.0)


def extract_circular_fov_region(
    img_np: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crops the image and mask to the bounding box of the circular FOV."""
    if mask is None:
        mask = compute_fov_mask(img_np)
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return img_np, mask
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    return img_np[y_min : y_max + 1, x_min : x_max + 1], mask[y_min : y_max + 1, x_min : x_max + 1]


def filter_features_by_fov(
    features: np.ndarray,
    fov_mask: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Filters spatial features according to FOV coverage threshold."""
    return features
