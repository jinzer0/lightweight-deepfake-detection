from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from pathlib import Path

import numpy as np
import pytest

from src.features.frequency import extract_frequency_features
from src.features.frequency_features import (
    compute_dct_spectrum,
    compute_fft_spectrum,
    extract_frequency_feature,
    image_to_grayscale_array,
    radial_average,
)
from src.visualization.radial_spectrum import save_radial_spectrum_plot
from src.visualization.spectrum import save_spectrum_image


def test_frequency_features_are_finite_with_expected_dimension(tiny_png: Path) -> None:
    features = extract_frequency_features(tiny_png)
    assert features.dtype == np.float32
    assert features.ndim == 1
    assert 100 <= features.shape[0] <= 250
    assert features.shape[0] == 220
    assert np.isfinite(features).all()


def test_target_frequency_feature_defaults_to_dct_with_radial_bins(tiny_png: Path) -> None:
    feature = extract_frequency_feature(tiny_png, {"image_size": 64})
    assert feature.dtype == np.float32
    assert feature.shape == (64,)
    assert np.isfinite(feature).all()


def test_target_frequency_feature_supports_fft_and_custom_bins(tiny_png: Path) -> None:
    feature = extract_frequency_feature(
        tiny_png,
        {"method": "fft", "image_size": 64, "radial_bins": 16, "log_scale": True, "normalize_feature": True},
    )
    assert feature.shape == (16,)
    assert np.isfinite(feature).all()


def test_target_frequency_feature_honors_nested_project_config(tiny_png: Path) -> None:
    nested_config = {
        "frequency": {
            "method": "fft",
            "image_size": 32,
            "radial_bins": 7,
            "log_scale": False,
            "normalize_feature": False,
        }
    }
    nested_feature = extract_frequency_feature(tiny_png, nested_config)
    flat_feature = extract_frequency_feature(tiny_png, nested_config["frequency"])
    default_feature = extract_frequency_feature(tiny_png, {"method": "fft", "image_size": 32})

    assert nested_feature.shape == (7,)
    assert np.isfinite(nested_feature).all()
    np.testing.assert_allclose(nested_feature, flat_feature)
    assert default_feature.shape == (64,)


def test_target_frequency_feature_supports_tensor_like_channel_first_input() -> None:
    rgb = np.zeros((3, 12, 10), dtype=np.float32)
    rgb[0, :, :] = 20.0
    rgb[1, :, :] = 40.0
    rgb[2, :, :] = 60.0
    gray = image_to_grayscale_array(rgb, image_size=8)
    feature = extract_frequency_feature(rgb, {"method": "dct", "image_size": 8, "radial_bins": 4})
    assert gray.shape == (8, 8)
    assert gray.dtype == np.float32
    assert feature.shape == (4,)
    assert np.isfinite(feature).all()


def test_spectrum_helpers_return_finite_2d_arrays(tiny_png: Path) -> None:
    gray = image_to_grayscale_array(tiny_png, image_size=32)
    dct_spectrum = compute_dct_spectrum(gray)
    fft_spectrum = compute_fft_spectrum(gray)
    radial = radial_average(np.log1p(dct_spectrum), bins=8)
    assert dct_spectrum.shape == (32, 32)
    assert fft_spectrum.shape == (32, 32)
    assert radial.shape == (8,)
    assert np.isfinite(dct_spectrum).all()
    assert np.isfinite(fft_spectrum).all()
    assert np.isfinite(radial).all()


def test_target_frequency_feature_rejects_unknown_method(tiny_png: Path) -> None:
    with pytest.raises(ValueError, match="Supported methods: dct, fft"):
        _feature = extract_frequency_feature(tiny_png, {"method": "wavelet"})


def test_visualization_helpers_save_png_files(tiny_png: Path, tmp_path: Path) -> None:
    spectrum_path = tmp_path / "figures" / "test_spectrum.png"
    radial_path = tmp_path / "figures" / "test_radial.png"
    feature = extract_frequency_feature(tiny_png, {"image_size": 64, "radial_bins": 16})

    returned_spectrum = Path(save_spectrum_image(tiny_png, str(spectrum_path), method="dct"))
    returned_radial = Path(save_radial_spectrum_plot(feature, str(radial_path)))

    assert returned_spectrum == spectrum_path
    assert returned_radial == radial_path
    assert spectrum_path.is_file()
    assert radial_path.is_file()
    assert spectrum_path.read_bytes().startswith(b"\x89PNG")
    assert radial_path.read_bytes().startswith(b"\x89PNG")
