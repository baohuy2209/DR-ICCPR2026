"""Fundus image preprocessing: circular ROI cropping and green channel enhancement."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


def crop_fundus_roi(
    img: np.ndarray,
    tolerance: int = 7,
) -> np.ndarray:
    """Automatically removes non-diagnostic black borders surrounding the retinal aperture.

    Args:
        img: Input image array in BGR or RGB format (H, W, 3) or grayscale (H, W).
        tolerance: Pixel intensity threshold below which pixels are treated as black background.

    Returns:
        Cropped image array bounded tightly around the retinal area.
    """
    if img is None or img.size == 0:
        return img

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    mask = gray > tolerance
    if not np.any(mask):
        return img

    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]

    cropped = img[ymin : ymax + 1, xmin : xmax + 1]
    return cropped if cropped.size > 0 else img


def extract_green_channel(
    img: np.ndarray,
    as_three_channel: bool = False,
) -> np.ndarray:
    """Extracts the green channel (index 1 in RGB) offering highest vascular/lesion contrast.

    Args:
        img: Image array of shape (H, W, 3).
        as_three_channel: If True, replicates green channel across all 3 channels.

    Returns:
        Green channel array (H, W) or (H, W, 3).
    """
    if img is None or img.ndim != 3:
        return img

    green = img[:, :, 1]
    if as_three_channel:
        return np.stack([green, green, green], axis=-1)
    return green


def apply_green_channel_ablation(
    img_rgb: np.ndarray,
    ablation_mode: str = "ben_graham_green",
) -> np.ndarray:
    """Applies fundus image preprocessing ablations centered on the green channel.

    Modes:
        - 'ben_graham_green': Gaussian background subtraction on bilateral-denoised green channel.
        - 'clahe_green': Contrast Limited Adaptive Histogram Equalization.
        - 'raw_green' (or fallback): Replicates raw green channel across 3 channels.

    Args:
        img_rgb: Input RGB image array of shape (H, W, 3).
        ablation_mode: 'ben_graham_green', 'clahe_green', or 'raw_green'.

    Returns:
        Processed 3-channel RGB image array of shape (H, W, 3).
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])
    elif ablation_mode == "clahe_green":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    else:
        processed = cv2.merge([g, g, g])

    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


def apply_ben_graham_enhancement(img_rgb: np.ndarray, sigma: int = 30) -> np.ndarray:
    """Applies classic Ben Graham Gaussian subtraction on RGB or green channel."""
    return apply_green_channel_ablation(img_rgb, ablation_mode="ben_graham_green")


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    return apply_green_channel_ablation(img, ablation_mode="clahe_green")


