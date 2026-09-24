"""Grad-CAM implementation for the global context CNN stream."""

from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Gradient-weighted Class Activation Mapping (Grad-CAM) for global CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def generate_saliency_map(
        self,
        img_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
        **model_kwargs: Any,
    ) -> np.ndarray:
        """Generates normalized [0, 1] Grad-CAM heatmap for the specified class.

        Args:
            img_tensor: Input image tensor of shape (1, 3, H, W).
            target_class: Target class index. If None, uses predicted class.
            target_size: Optional (H, W) to upsample heatmap to.

        Returns:
            2D numpy array of shape (H, W) with values in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        outputs = self.model(img_tensor, **model_kwargs)
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("probs"))
        else:
            logits = outputs

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture
            h, w = target_size if target_size else (img_tensor.shape[2], img_tensor.shape[3])
            return np.zeros((h, w), dtype=np.float32)

        # Global average pool the gradients over spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H_act, W_act)
        cam = F.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max > c_min:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        if target_size is not None and cam_np.shape != target_size:
            cam_np = cv2.resize(cam_np, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        return np.clip(cam_np, 0.0, 1.0).astype(np.float32)
