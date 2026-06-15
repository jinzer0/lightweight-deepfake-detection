from __future__ import annotations

from PIL import Image

from scripts.robustness_test import (
    CorruptionSpec,
    apply_center_crop_resize,
    apply_corruption,
    apply_gaussian_blur,
    apply_jpeg,
    apply_resize_down_up,
)


def test_corruptions_return_rgb_pil_images_with_original_size() -> None:
    image = Image.new("RGB", (37, 29), color=(20, 80, 140))
    outputs = [
        apply_corruption(image, CorruptionSpec("clean", "clean", "none")),
        apply_jpeg(image, 95),
        apply_jpeg(image, 75),
        apply_resize_down_up(image, 0.5),
        apply_resize_down_up(image, 0.25),
        apply_gaussian_blur(image, 1.0),
        apply_gaussian_blur(image, 2.0),
        apply_center_crop_resize(image),
    ]

    for output in outputs:
        assert isinstance(output, Image.Image)
        assert output.mode == "RGB"
        assert output.size == image.size


def test_corruptions_do_not_mutate_source_image() -> None:
    image = Image.new("RGB", (31, 31), color=(10, 30, 50))
    before = image.tobytes()

    _ = apply_resize_down_up(image, 0.5)
    _ = apply_center_crop_resize(image)
    _ = apply_gaussian_blur(image, 1.0)

    assert image.tobytes() == before
