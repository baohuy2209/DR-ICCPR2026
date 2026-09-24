"""Augmentation and transformation pipelines for global and local patch streams."""

from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np


def get_global_augmentations(img_size: int = 512, is_train: bool = True) -> A.Compose:
    """Constructs the Albumentations transformation pipeline for the Global Context branch.

    Args:
        img_size: Target spatial resolution (width and height) in pixels.
        is_train: If True, applies randomized spatial and photometric augmentations;
            otherwise, applies deterministic bilinear resize only.

    Returns:
        Albumentations Compose transform pipeline.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-30, 30),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.10,
                    contrast_limit=0.10,
                    p=0.5,
                ),
            ]
        )

    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
        ]
    )


def get_patch_augmentations(patch_size: int = 128, is_train: bool = True) -> Optional[A.Compose]:
    """Constructs patch-level augmentations preserving micro-lesion morphological details."""
    if not is_train:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )


def get_eval_transforms(img_size: int = 512) -> A.Compose:
    """Returns deterministic evaluation transforms for global image."""
    return get_global_augmentations(img_size=img_size, is_train=False)


class Denormalize:
    """Inverts ImageNet normalization for visualization."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, tensor_or_img: np.ndarray) -> np.ndarray:
        """Denormalizes an image array from [-mean/std] back to [0, 255]."""
        img = tensor_or_img * self.std + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img


class DualBranchTransform:
    """Applies combined global and patch branch transformations."""

    def __init__(self, global_size: int = 512, patch_size: int = 128, is_train: bool = True) -> None:
        self.global_transform = get_global_augmentations(img_size=global_size, is_train=is_train)
        self.patch_transform = get_patch_augmentations(patch_size=patch_size, is_train=is_train)

    def transform_global(self, img: np.ndarray) -> np.ndarray:
        return self.global_transform(image=img)["image"]

    def transform_patch(self, patch: np.ndarray) -> np.ndarray:
        if self.patch_transform is not None:
            return self.patch_transform(image=patch)["image"]
        return patch

