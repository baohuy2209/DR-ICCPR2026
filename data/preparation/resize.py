"""Image resizing utilities for global context representations (e.g. 512x512)."""

from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image


def resize_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    output_path: Optional[Union[str, Path]] = None,
    resample: int = Image.Resampling.LANCZOS,
) -> Path:
    """Resizes an image file to target dimensions and optionally saves to output_path.

    Args:
        image_path: Source image path.
        target_size: (width, height) tuple, default (512, 512).
        output_path: Destination path. If None, saves next to input as '<stem>_512.jpg'.
        resample: PIL resampling filter (default: LANCZOS).

    Returns:
        Path to the saved resized image.
    """
    in_p = Path(image_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Image not found at: {in_p}")

    out_p = Path(output_path) if output_path is not None else in_p.parent / f"{in_p.stem}_{target_size[0]}.jpg"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(in_p) as img:
        img_rgb = img.convert("RGB")
        resized = img_rgb.resize(target_size, resample=resample)
        resized.save(out_p, quality=95)

    return out_p
