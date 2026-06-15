from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportOptionalSubscript=false

import sys
from pathlib import Path

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_rgb_image(path: Path, color: tuple[int, int, int], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (19, 23), color=color)
    pixels = image.load()
    assert pixels is not None
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (
                (color[0] + x * 7 + y * 3) % 256,
                (color[1] + x * 5 + y * 11) % 256,
                (color[2] + x * 13 + y * 17) % 256,
            )
    image.save(path, format=fmt)


@pytest.fixture()
def synthetic_real_fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "real_fake_images"
    for class_name, base_color in {"REAL": (35, 80, 125), "FAKE": (150, 70, 25)}.items():
        for index in range(6):
            fmt = "PNG" if index % 2 == 0 else "JPEG"
            suffix = "png" if fmt == "PNG" else "jpg"
            color = tuple((component + index * 23) % 256 for component in base_color)
            _write_rgb_image(root / class_name / f"{class_name.lower()}_{index:02d}.{suffix}", color, fmt)
    return root


@pytest.fixture()
def tiny_png(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.png"
    _write_rgb_image(path, (20, 40, 60), "PNG")
    return path


@pytest.fixture()
def tiny_jpg(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.jpg"
    _write_rgb_image(path, (80, 120, 160), "JPEG")
    return path
