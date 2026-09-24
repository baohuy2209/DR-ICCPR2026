"""Reproducibility utilities for random seed management across libraries."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch CUDA backend for determinism.
            Note that exact bitwise reproducibility across different GPU hardware
            architectures is not guaranteed due to non-deterministic atomic CUDA kernels.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initialization function for reproducible multi-threaded loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
