"""Dynamic MIL bag collate functions for DataLoader batching."""

from typing import Any, Dict, List

import torch


def trainer_compatible_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of FundusDualBranchDataset sample dictionaries into a batched dictionary.

    Handles variable numbers of local patches per sample by padding bags up to
    the maximum patch count (K_max) within the batch and generating a boolean patch_mask.

    Args:
        batch: List of dictionaries produced by FundusDualBranchDataset.__getitem__.

    Returns:
        Dict containing:
            - 'global_img': Tensor of shape [B, 3, H, W]
            - 'local_patches': Tensor of shape [B, K_max, 3, H_p, W_p]
            - 'patch_mask': Bool tensor of shape [B, K_max] (True = valid, False = padded)
            - 'label': Int64 tensor of shape [B]
            - 'image_id': List of strings of length B
            - 'patient_id': List of strings of length B
            - 'dataset_key': List of strings of length B
    """
    b = len(batch)
    global_imgs = torch.stack([item["global_img"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    # Determine maximum number of patches in this batch
    patch_counts = [item["local_patches"].shape[0] for item in batch]
    k_max = max(patch_counts)
    _, c, hp, wp = batch[0]["local_patches"].shape

    padded_patches = torch.zeros(b, k_max, c, hp, wp, dtype=batch[0]["local_patches"].dtype)
    patch_mask = torch.zeros(b, k_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        k = item["local_patches"].shape[0]
        if k > 0:
            padded_patches[i, :k] = item["local_patches"]
            patch_mask[i, :k] = True

    return {
        "global_img": global_imgs,
        "local_patches": padded_patches,
        "patch_mask": patch_mask,
        "label": labels,
        "image_id": [item["image_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "dataset_key": [item["dataset_key"] for item in batch],
    }
