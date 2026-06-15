from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from pathlib import Path

import numpy as np
import pytest

from src.utils.image_io import CorruptImageError, load_rgb_image, pad_to_multiple, prepare_frequency_image


def test_load_rgb_image_converts_jpg_and_png(tiny_jpg: Path, tiny_png: Path) -> None:
    jpg_image = load_rgb_image(tiny_jpg)
    png_image = load_rgb_image(tiny_png)
    assert jpg_image.mode == "RGB"
    assert png_image.mode == "RGB"
    assert jpg_image.size == (19, 23)
    assert png_image.size == (19, 23)


def test_prepare_frequency_image_outputs_512_luminance(tiny_png: Path) -> None:
    luminance = prepare_frequency_image(tiny_png)
    assert luminance.shape == (512, 512)
    assert luminance.dtype == np.float32
    assert np.isfinite(luminance).all()


def test_pad_to_multiple_uses_edge_padding() -> None:
    array = np.arange(15, dtype=np.float32).reshape(3, 5)
    padded = pad_to_multiple(array, multiple=8)
    assert padded.shape == (8, 8)
    assert np.array_equal(padded[:3, :5], array)
    assert padded[-1, -1] == array[-1, -1]


def test_load_rgb_image_rejects_corrupt_image(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.png"
    _ = corrupt_path.write_bytes(b"not an image")
    with pytest.raises(CorruptImageError):
        load_rgb_image(corrupt_path)
