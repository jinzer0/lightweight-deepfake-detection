from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportExplicitAny=false

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.utils.image_io import DEFAULT_FREQUENCY_IMAGE_SIZE, LUMINANCE_WEIGHTS, prepare_frequency_image

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("Pillow is required for frequency feature extraction; install requirements.txt.") from exc

try:
    from scipy.fft import dct as _imported_scipy_dct
except ImportError:
    _scipy_dct = None
else:
    _scipy_dct = cast(Callable[..., np.ndarray], _imported_scipy_dct)


FEATURE_DTYPE = np.float32
DEFAULT_METHOD = "dct"
DEFAULT_RADIAL_BINS = 64
SUPPORTED_METHODS = ("dct", "fft")


def image_to_grayscale_array(image: Any, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if isinstance(image, (str, Path)):
        return prepare_frequency_image(image, image_size=image_size).astype(FEATURE_DTYPE, copy=False)

    if _is_pil_image(image):
        return prepare_frequency_image(image, image_size=image_size).astype(FEATURE_DTYPE, copy=False)

    array = _to_numpy_array(image)
    grayscale = _array_to_grayscale(array)
    if grayscale.shape != (int(image_size), int(image_size)):
        grayscale = _resize_grayscale(grayscale, image_size)
    return grayscale.astype(FEATURE_DTYPE, copy=False)


def compute_dct_spectrum(gray: np.ndarray) -> np.ndarray:
    array = _validate_gray(gray)
    if _scipy_dct is not None:
        spectrum = _scipy_dct(_scipy_dct(array, axis=0, norm="ortho", type=2), axis=1, norm="ortho", type=2)
        return np.abs(spectrum).astype(FEATURE_DTYPE, copy=False)

    matrix_y = _dct_matrix(array.shape[0])
    matrix_x = _dct_matrix(array.shape[1])
    return np.abs(matrix_y @ array @ matrix_x.T).astype(FEATURE_DTYPE, copy=False)


def compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    array = _validate_gray(gray)
    centered = np.fft.fftshift(np.fft.fft2(array))
    return np.abs(centered).astype(FEATURE_DTYPE, copy=False)


def radial_average(spectrum: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("bins must be positive")

    values = _validate_gray(spectrum)
    radius = _normalized_radius(values.shape)
    bin_edges = np.linspace(0.0, 1.0, int(bins) + 1, dtype=FEATURE_DTYPE)
    profile = np.empty(int(bins), dtype=FEATURE_DTYPE)
    flat_values = values.reshape(-1)
    flat_radius = radius.reshape(-1)

    for index in range(int(bins)):
        if index == int(bins) - 1:
            mask = (flat_radius >= bin_edges[index]) & (flat_radius <= bin_edges[index + 1])
        else:
            mask = (flat_radius >= bin_edges[index]) & (flat_radius < bin_edges[index + 1])
        profile[index] = float(np.mean(flat_values[mask])) if np.any(mask) else 0.0

    return profile.astype(FEATURE_DTYPE, copy=False)


def extract_frequency_feature(image: Any, config: Mapping[str, Any] | None = None) -> np.ndarray:
    settings = _frequency_settings(config)
    method = str(settings.get("method", DEFAULT_METHOD)).lower()
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unsupported frequency method '{method}'. Supported methods: {supported}.")

    image_size = int(settings.get("image_size", DEFAULT_FREQUENCY_IMAGE_SIZE))
    radial_bins = int(settings.get("radial_bins", DEFAULT_RADIAL_BINS))
    log_scale = bool(settings.get("log_scale", True))
    normalize_feature = bool(settings.get("normalize_feature", settings.get("normalize", True)))

    gray = image_to_grayscale_array(image, image_size=image_size)
    spectrum = compute_dct_spectrum(gray) if method == "dct" else compute_fft_spectrum(gray)
    if log_scale:
        spectrum = np.log1p(spectrum).astype(FEATURE_DTYPE, copy=False)

    feature = radial_average(spectrum, radial_bins)
    if normalize_feature:
        feature = _normalize(feature)

    if feature.shape != (radial_bins,):
        raise ValueError(f"frequency feature shape must be ({radial_bins},), got {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError("frequency feature must contain only finite values")
    return feature.astype(FEATURE_DTYPE, copy=False)


def _frequency_settings(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    frequency_config = config.get("frequency")
    if isinstance(frequency_config, Mapping):
        return frequency_config
    return config


def _is_pil_image(value: Any) -> bool:
    return isinstance(value, Image.Image)


def _to_numpy_array(image: Any) -> np.ndarray:
    if _looks_like_torch_tensor(image):
        image = image.detach().cpu().numpy()
    return np.asarray(image)


def _looks_like_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def _array_to_grayscale(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(FEATURE_DTYPE, copy=False)
    if array.ndim != 3:
        raise ValueError(f"image array must be 2D grayscale or 3D RGB-like, got shape {array.shape}")

    channel_last = _channel_last(array)
    if channel_last.shape[2] == 1:
        return channel_last[:, :, 0].astype(FEATURE_DTYPE, copy=False)
    if channel_last.shape[2] < 3:
        raise ValueError(f"RGB-like image array must have at least 3 channels, got shape {array.shape}")
    rgb = channel_last[:, :, :3].astype(FEATURE_DTYPE, copy=False)
    return np.tensordot(rgb, LUMINANCE_WEIGHTS, axes=([-1], [0])).astype(FEATURE_DTYPE, copy=False)


def _channel_last(array: np.ndarray) -> np.ndarray:
    if array.shape[-1] in (1, 3, 4):
        return array
    if array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1)
    return array


def _resize_grayscale(gray: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(gray.astype(FEATURE_DTYPE, copy=False), mode="F")
    resized = image.resize((int(image_size), int(image_size)), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=FEATURE_DTYPE)


def _validate_gray(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=FEATURE_DTYPE)
    if values.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("array must contain only finite values")
    return values


def _normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y_indices, x_indices = np.indices((height, width), dtype=FEATURE_DTYPE)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)
    max_radius = float(radius.max())
    return radius / max_radius if max_radius > 0.0 else radius


def _normalize(feature: np.ndarray) -> np.ndarray:
    values = feature.astype(FEATURE_DTYPE, copy=True)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std > 0.0:
        return ((values - mean) / std).astype(FEATURE_DTYPE, copy=False)
    return (values - mean).astype(FEATURE_DTYPE, copy=False)


def _dct_matrix(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=FEATURE_DTYPE)
    coefficients = np.arange(size, dtype=FEATURE_DTYPE)[:, None]
    matrix = np.cos(np.pi * (indices + 0.5) * coefficients / float(size)).astype(FEATURE_DTYPE)
    matrix[0, :] *= np.sqrt(1.0 / float(size))
    if size > 1:
        matrix[1:, :] *= np.sqrt(2.0 / float(size))
    return matrix
