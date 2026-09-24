"""Hardware device detection and mixed-precision helpers."""

from typing import Tuple
import torch


def resolve_device(requested_device: str = "auto") -> torch.device:
    """Resolve compute device based on hardware availability and user request."""
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def resolve_precision(device: torch.device, requested_amp: str = "auto") -> Tuple[bool, torch.dtype]:
    """Resolve AMP enablement and dtype based on device capability.

    Args:
        device: Target compute device.
        requested_amp: One of 'auto', 'fp16', 'bf16', 'none'.

    Returns:
        enabled: Boolean indicating if torch.cuda.amp.autocast should be used.
        dtype: Target torch.dtype (float16, bfloat16, or float32).
    """
    if device.type != "cuda" or requested_amp == "none":
        return False, torch.float32

    if requested_amp == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16

    if requested_amp == "fp16":
        return True, torch.float16

    # 'auto' mode: prefer bf16 on Ampere or newer GPUs (Compute Capability >= 8.0)
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def get_device_and_amp(requested_device: str = "auto", requested_amp: str = "auto") -> Tuple[torch.device, bool]:
    """Convenience helper returning resolved device and boolean AMP flag."""
    dev = resolve_device(requested_device)
    amp_enabled, _ = resolve_precision(dev, requested_amp)
    return dev, amp_enabled

