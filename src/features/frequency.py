from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportExplicitAny=false, reportAny=false, reportAttributeAccessIssue=false, reportArgumentType=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.image_io import DEFAULT_FREQUENCY_IMAGE_SIZE, DCT_BLOCK_SIZE, pad_to_multiple, prepare_frequency_image

try:
    from scipy.fft import dct as _scipy_dct
except ImportError:  # pragma: no cover - scipy is declared in requirements.txt
    _scipy_dct = None

DEFAULT_RADIAL_BINS = 64
DEFAULT_FFT_EPSILON = 1e-8
FEATURE_DTYPE = np.float32
DCT_POLICY = "dct_ii_ortho_whole_image_and_8x8_blocks_edge_pad"
DCT_BACKEND = "scipy.fft.dct" if _scipy_dct is not None else "numpy_orthonormal_dct_ii_matrix"


@dataclass(frozen=True)
class FrequencyFeatureConfig:
    image_size: int = DEFAULT_FREQUENCY_IMAGE_SIZE
    radial_bins: int = DEFAULT_RADIAL_BINS
    fft_epsilon: float = DEFAULT_FFT_EPSILON

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "image_size": int(self.image_size),
            "radial_bins": int(self.radial_bins),
            "fft_epsilon": float(self.fft_epsilon),
            "dct_policy": DCT_POLICY,
            "dct_backend": DCT_BACKEND,
        }


def extract_frequency_features(
    image: Any | str | Path | np.ndarray,
    *,
    image_size: int = DEFAULT_FREQUENCY_IMAGE_SIZE,
    radial_bins: int = DEFAULT_RADIAL_BINS,
    fft_epsilon: float = DEFAULT_FFT_EPSILON,
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        luminance = np.asarray(image, dtype=np.float32)
        if luminance.ndim != 2:
            raise ValueError(f"array image must be 2D luminance, got shape {luminance.shape}")
    else:
        luminance = prepare_frequency_image(image, image_size=image_size)

    features = np.concatenate(
        [
            _fft_features(luminance, radial_bins=radial_bins, epsilon=fft_epsilon),
            _dct_features(luminance),
        ]
    ).astype(FEATURE_DTYPE, copy=False)

    if features.ndim != 1 or not np.isfinite(features).all():
        raise ValueError("frequency features must be a finite 1D vector")
    if not 100 <= int(features.shape[0]) <= 250:
        raise ValueError(f"frequency feature dimension must be between 100 and 250, got {features.shape[0]}")
    return features


def extract_frequency_feature_batch(
    images: list[Any | str | Path | np.ndarray],
    *,
    image_size: int = DEFAULT_FREQUENCY_IMAGE_SIZE,
    radial_bins: int = DEFAULT_RADIAL_BINS,
    fft_epsilon: float = DEFAULT_FFT_EPSILON,
) -> np.ndarray:
    rows = [
        extract_frequency_features(image, image_size=image_size, radial_bins=radial_bins, fft_epsilon=fft_epsilon)
        for image in images
    ]
    return np.vstack(rows).astype(FEATURE_DTYPE, copy=False)


def frequency_feature_dim(radial_bins: int = DEFAULT_RADIAL_BINS) -> int:
    return int(radial_bins) + 156


def _fft_features(luminance: np.ndarray, *, radial_bins: int, epsilon: float) -> np.ndarray:
    centered = np.fft.fftshift(np.fft.fft2(luminance.astype(np.float32, copy=False)))
    magnitude = np.abs(centered)
    log_spectrum = np.log(magnitude + float(epsilon)).astype(np.float32)
    radial_profile = _radial_profile(log_spectrum, radial_bins)

    radius = _normalized_radius(log_spectrum.shape)
    low_mask = radius < 1.0 / 3.0
    mid_mask = (radius >= 1.0 / 3.0) & (radius < 2.0 / 3.0)
    high_mask = radius >= 2.0 / 3.0
    low_energy = _mean_masked(log_spectrum, low_mask)
    mid_energy = _mean_masked(log_spectrum, mid_mask)
    high_energy = _mean_masked(log_spectrum, high_mask)
    total_energy = low_energy + mid_energy + high_energy
    high_ratio = high_energy / (total_energy + epsilon)
    low_high_ratio = low_energy / (high_energy + epsilon)
    slope = _linear_slope(np.linspace(0.0, 1.0, int(radial_bins), dtype=np.float32), radial_profile)

    summary = np.asarray([low_energy, mid_energy, high_energy, high_ratio, low_high_ratio, slope], dtype=np.float32)
    return np.concatenate([radial_profile, summary])


def _dct_features(luminance: np.ndarray) -> np.ndarray:
    padded = pad_to_multiple(luminance.astype(np.float32, copy=False), DCT_BLOCK_SIZE)
    whole = _dct2(padded)
    whole_abs = np.abs(whole)
    whole_low, whole_mid, whole_high = _coefficient_band_means(whole_abs)
    whole_total = whole_low + whole_mid + whole_high
    whole_stats = np.asarray(
        [
            whole_low,
            whole_mid,
            whole_high,
            whole_high / (whole_total + DEFAULT_FFT_EPSILON),
            whole_low / (whole_high + DEFAULT_FFT_EPSILON),
            float(np.mean(whole)),
            float(np.std(whole)),
            float(np.max(whole)),
            float(np.mean(whole_abs)),
            float(np.std(whole_abs)),
        ],
        dtype=np.float32,
    )

    blocks = _dct_blocks(padded)
    block_abs = np.abs(blocks)
    per_coeff_mean = block_abs.mean(axis=0).reshape(-1)
    per_coeff_std = block_abs.std(axis=0).reshape(-1)

    block_low, block_mid, block_high = _block_band_means(block_abs)
    block_total = block_low + block_mid + block_high
    block_high_ratio = block_high / (block_total + DEFAULT_FFT_EPSILON)
    block_summaries = np.asarray(
        [
            float(np.mean(block_low)),
            float(np.std(block_low)),
            float(np.mean(block_mid)),
            float(np.std(block_mid)),
            float(np.mean(block_high)),
            float(np.std(block_high)),
            float(np.mean(block_high_ratio)),
            float(np.std(block_high_ratio)),
            float(np.mean(blocks)),
            float(np.std(blocks)),
            float(np.max(blocks)),
            float(np.mean(block_abs)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([whole_stats, per_coeff_mean, per_coeff_std, block_summaries]).astype(np.float32, copy=False)


def _radial_profile(values: np.ndarray, radial_bins: int) -> np.ndarray:
    if radial_bins <= 0:
        raise ValueError("radial_bins must be positive")
    radius = _normalized_radius(values.shape)
    bin_edges = np.linspace(0.0, 1.0, int(radial_bins) + 1, dtype=np.float32)
    profile = np.empty(int(radial_bins), dtype=np.float32)
    flat_values = values.reshape(-1)
    flat_radius = radius.reshape(-1)
    for index in range(int(radial_bins)):
        if index == int(radial_bins) - 1:
            mask = (flat_radius >= bin_edges[index]) & (flat_radius <= bin_edges[index + 1])
        else:
            mask = (flat_radius >= bin_edges[index]) & (flat_radius < bin_edges[index + 1])
        profile[index] = float(np.mean(flat_values[mask])) if np.any(mask) else 0.0
    return profile


def _normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.indices((height, width), dtype=np.float32)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    max_radius = float(radius.max())
    return radius / max_radius if max_radius > 0.0 else radius


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def _linear_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    centered_x = x_values - float(np.mean(x_values))
    centered_y = y_values - float(np.mean(y_values))
    denominator = float(np.sum(centered_x * centered_x))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(centered_x * centered_y) / denominator)


def _dct2(array: np.ndarray) -> np.ndarray:
    if _scipy_dct is not None:
        return _scipy_dct(_scipy_dct(array, axis=0, norm="ortho", type=2), axis=1, norm="ortho", type=2).astype(np.float32)
    matrix_y = _dct_matrix(array.shape[0])
    matrix_x = _dct_matrix(array.shape[1])
    return (matrix_y @ array @ matrix_x.T).astype(np.float32)


def _dct_matrix(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=np.float32)
    coeffs = np.arange(size, dtype=np.float32)[:, None]
    matrix = np.cos(np.pi * (indices + 0.5) * coeffs / float(size)).astype(np.float32)
    matrix[0, :] *= np.sqrt(1.0 / float(size))
    if size > 1:
        matrix[1:, :] *= np.sqrt(2.0 / float(size))
    return matrix


def _dct_blocks(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    reshaped = array.reshape(height // DCT_BLOCK_SIZE, DCT_BLOCK_SIZE, width // DCT_BLOCK_SIZE, DCT_BLOCK_SIZE)
    blocks = reshaped.transpose(0, 2, 1, 3).reshape(-1, DCT_BLOCK_SIZE, DCT_BLOCK_SIZE)
    return np.stack([_dct2(block) for block in blocks], axis=0).astype(np.float32)


def _coefficient_band_means(coefficients: np.ndarray) -> tuple[float, float, float]:
    y, x = np.indices(coefficients.shape, dtype=np.int32)
    normalized = (x + y).astype(np.float32) / float(sum(coefficients.shape) - 2)
    low = _mean_masked(coefficients, normalized < 1.0 / 3.0)
    mid = _mean_masked(coefficients, (normalized >= 1.0 / 3.0) & (normalized < 2.0 / 3.0))
    high = _mean_masked(coefficients, normalized >= 2.0 / 3.0)
    return low, mid, high


def _block_band_means(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((DCT_BLOCK_SIZE, DCT_BLOCK_SIZE), dtype=np.int32)
    coordinate_sum = x + y
    low_mask = coordinate_sum <= 3
    mid_mask = (coordinate_sum > 3) & (coordinate_sum <= 8)
    high_mask = coordinate_sum > 8
    return blocks[:, low_mask].mean(axis=1), blocks[:, mid_mask].mean(axis=1), blocks[:, high_mask].mean(axis=1)
