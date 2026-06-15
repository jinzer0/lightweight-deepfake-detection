from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false

from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Pillow is required for src.utils.image_io; install dependencies with `pip install -r requirements.txt`.") from exc


DEFAULT_FREQUENCY_IMAGE_SIZE = 512
DCT_BLOCK_SIZE = 8
LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class CorruptImageError(RuntimeError):
    pass


def load_rgb_image(path: str | Path) -> Any:
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
            rgb_image.load()
            return rgb_image
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CorruptImageError(f"Failed to decode image: {image_path}") from exc


def image_size(path: str | Path) -> tuple[int, int]:
    return load_rgb_image(path).size


def to_luminance_array(image: Any) -> np.ndarray:
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.float32)
    return np.tensordot(rgb_array, LUMINANCE_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def prepare_frequency_image(image: Any | str | Path, image_size: int = DEFAULT_FREQUENCY_IMAGE_SIZE) -> np.ndarray:
    rgb_image = load_rgb_image(image) if isinstance(image, (str, Path)) else ImageOps.exif_transpose(image).convert("RGB")
    resized_image = rgb_image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    return to_luminance_array(resized_image)


def pad_to_multiple(array: np.ndarray, multiple: int = DCT_BLOCK_SIZE) -> np.ndarray:
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    np_array = np.asarray(array)
    if np_array.ndim < 2:
        raise ValueError("array must have at least two dimensions")

    height, width = np_array.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return np_array

    pad_widths = [(0, 0)] * (np_array.ndim - 2) + [(0, pad_height), (0, pad_width)]
    return np.pad(np_array, pad_widths, mode="edge")
