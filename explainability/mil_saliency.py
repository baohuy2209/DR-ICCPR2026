"""Local Multi-Instance Learning (MIL) Attention Saliency projection."""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class MILAttentionSaliency:
    """Projects patch-level attention weights into a spatial fundus saliency heatmap."""

    def __init__(self, patch_size: int = 128, sigma_smooth: float = 0.0) -> None:
        self.patch_size = patch_size
        self.sigma_smooth = sigma_smooth

    def generate_saliency_map(
        self,
        attention_weights: Union[np.ndarray, torch.Tensor],
        patch_anchors: Union[np.ndarray, torch.Tensor],
        img_shape: Tuple[int, int],
        normalize: bool = True,
    ) -> np.ndarray:
        """Projects scalar attention weights into 2D full-image heatmap.

        Args:
            attention_weights: (K,) or (1, K) array of non-negative attention weights summing to 1.
            patch_anchors: (K, 2) array of integer coordinates [y0, x0].
            img_shape: (H, W) destination spatial resolution.
            normalize: If True, normalizes final heatmap to [0.0, 1.0].

        Returns:
            2D numpy array of shape (H, W) in float32.
        """
        if isinstance(attention_weights, torch.Tensor):
            weights = attention_weights.detach().cpu().squeeze().numpy()
        else:
            weights = np.asarray(attention_weights, dtype=np.float32).squeeze()

        if isinstance(patch_anchors, torch.Tensor):
            anchors = patch_anchors.detach().cpu().numpy()
        else:
            anchors = np.asarray(patch_anchors, dtype=np.int32)

        h, w = img_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        p = self.patch_size

        if weights.ndim == 0:
            weights = np.array([float(weights)])

        for idx, (y0, x0) in enumerate(anchors):
            if idx >= len(weights):
                break
            y1 = min(y0 + p, h)
            x1 = min(x0 + p, w)
            if y1 > y0 and x1 > x0:
                w_val = float(weights[idx])
                heatmap[y0:y1, x0:x1] += w_val
                count_map[y0:y1, x0:x1] += 1.0

        # Average over overlapping patch coverage
        valid_mask = count_map > 0
        heatmap[valid_mask] /= count_map[valid_mask]

        if self.sigma_smooth > 0:
            ksize = int(self.sigma_smooth * 4) + 1
            if ksize % 2 == 0:
                ksize += 1
            heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), self.sigma_smooth)

        if normalize:
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)
