from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

try:
    from scipy.fft import dct  # pyright: ignore[reportMissingTypeStubs]
except Exception:  # pragma: no cover
    dct = None

FEATURE_DTYPE = np.float32
DEFAULT_RADIAL_BINS = 64
DEFAULT_IMAGE_SIZE = 512
DEFAULT_FREQUENCY_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
BLOCK_SIZE = 8
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def image_to_grayscale_array(image: Any, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        pil = image.convert("L")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        pil = Image.fromarray(np.asarray(arr).astype(np.uint8), mode="L")
    pil = pil.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr.astype(FEATURE_DTYPE, copy=False)


def compute_dct_spectrum(gray: np.ndarray) -> np.ndarray:
    arr = _validate_gray(gray)
    if dct is not None:
        first = dct(arr, axis=0, norm="ortho", type=2)  # pyright: ignore[reportCallIssue]
        coeff = dct(first, axis=1, norm="ortho", type=2)  # pyright: ignore[reportCallIssue]
    else:
        coeff = np.fft.fft2(arr).real
    return np.abs(np.asarray(coeff)).astype(FEATURE_DTYPE, copy=False)


def compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    arr = _validate_gray(gray)
    return np.abs(np.fft.fftshift(np.fft.fft2(arr))).astype(FEATURE_DTYPE, copy=False)


def radial_average(spectrum: np.ndarray, bins: int = DEFAULT_RADIAL_BINS) -> np.ndarray:
    arr = _validate_gray(spectrum)
    yy, xx = np.indices(arr.shape)
    center_y = (arr.shape[0] - 1) / 2.0
    center_x = (arr.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    radius = radius / max(radius.max(), 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = np.zeros(bins, dtype=np.float32)
    for idx in range(bins):
        mask = (radius >= edges[idx]) & (radius < edges[idx + 1] if idx < bins - 1 else radius <= edges[idx + 1])
        out[idx] = float(arr[mask].mean()) if np.any(mask) else 0.0
    return out


def high_frequency_ratio(spectrum: np.ndarray, cutoff: float = 0.5) -> float:
    arr = _validate_gray(spectrum)
    radius = _radius(arr.shape)
    energy = np.square(arr.astype(np.float64))
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    return float(energy[radius >= cutoff].sum() / total)


def band_energy_ratios(spectrum: np.ndarray) -> np.ndarray:
    arr = _validate_gray(spectrum)
    radius = _radius(arr.shape)
    energy = np.square(arr.astype(np.float64))
    total = float(energy.sum())
    if total <= 0.0:
        return np.zeros(3, dtype=np.float32)
    bands = [radius < 0.25, (radius >= 0.25) & (radius < 0.5), radius >= 0.5]
    return np.asarray([energy[band].sum() / total for band in bands], dtype=np.float32)


def block_dct_statistics(gray: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    arr = _validate_gray(gray)
    stats: list[float] = []
    h = arr.shape[0] - arr.shape[0] % block_size
    w = arr.shape[1] - arr.shape[1] % block_size
    blocks = arr[:h, :w].reshape(h // block_size, block_size, w // block_size, block_size).swapaxes(1, 2)
    coeffs = []
    for block in blocks.reshape(-1, block_size, block_size):
        coeffs.append(compute_dct_spectrum(block).reshape(-1))
    values = np.concatenate(coeffs).astype(np.float32) if coeffs else np.zeros(1, dtype=np.float32)
    stats.extend([float(values.mean()), float(values.std()), float(values.max())])
    stats.extend(float(np.quantile(values, q)) for q in QUANTILES)
    return np.asarray(stats, dtype=np.float32)


def extract_frequency_feature_dict(image: Any, image_size: int = DEFAULT_IMAGE_SIZE, radial_bins: int = DEFAULT_RADIAL_BINS) -> dict[str, Any]:
    gray = image_to_grayscale_array(image, image_size=image_size)
    dct_spec = compute_dct_spectrum(gray)
    fft_spec = compute_fft_spectrum(gray)
    dct_radial = radial_average(dct_spec, bins=radial_bins)
    fft_radial = radial_average(fft_spec, bins=radial_bins)
    hfr = high_frequency_ratio(fft_spec)
    bands = band_energy_ratios(fft_spec)
    block_stats = block_dct_statistics(gray)
    feature = np.concatenate([dct_radial, fft_radial, np.asarray([hfr], dtype=np.float32), bands, block_stats]).astype(np.float32)
    if not np.isfinite(feature).all():
        raise ValueError("frequency feature contains NaN or Inf")
    return {
        "feature": feature,
        "dct_radial_spectrum": dct_radial,
        "fft_radial_spectrum": fft_radial,
        "high_frequency_ratio": hfr,
        "band_energy_ratio": bands,
        "block_dct_statistics": block_stats,
    }



def extract_frequency_feature(image: Any, config: dict[str, Any] | None = None) -> np.ndarray:
    """Backward-compatible radial spectrum extractor used by legacy tests."""
    cfg = dict((config or {}).get("frequency", config or {}))
    image_size = int(cfg.get("image_size", DEFAULT_IMAGE_SIZE))
    bins = int(cfg.get("radial_bins", DEFAULT_RADIAL_BINS))
    method = str(cfg.get("method", "dct"))
    if method not in {"dct", "fft"}:
        raise ValueError("Supported methods: dct, fft")
    gray = image_to_grayscale_array(image, image_size=image_size)
    spectrum = compute_fft_spectrum(gray) if method == "fft" else compute_dct_spectrum(gray)
    feature = radial_average(spectrum, bins=bins)
    if bool(cfg.get("log_scale", False)):
        feature = np.log1p(feature).astype(np.float32)
    if bool(cfg.get("normalize_feature", False)):
        norm = float(np.linalg.norm(feature))
        if norm > 0.0:
            feature = (feature / norm).astype(np.float32)
    return feature.astype(np.float32, copy=False)


def extract_frequency_features(image: Any, image_size: int = DEFAULT_IMAGE_SIZE, radial_bins: int = DEFAULT_RADIAL_BINS) -> np.ndarray:
    return extract_frequency_feature_dict(image, image_size=image_size, radial_bins=radial_bins)["feature"]


def expected_feature_dim(radial_bins: int = DEFAULT_RADIAL_BINS) -> int:
    return radial_bins * 2 + 1 + 3 + 3 + len(QUANTILES)


def _validate_gray(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grayscale array, got shape {arr.shape}")
    return arr


def _radius(shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.indices(shape)
    center_y = (shape[0] - 1) / 2.0
    center_x = (shape[1] - 1) / 2.0
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    return radius / max(radius.max(), 1.0)




def _frequency_settings(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    frequency = config.get("frequency")
    if isinstance(frequency, Mapping):
        return frequency
    return config


def _normalized_radius(shape: tuple[int, int]) -> np.ndarray:
    return _radius(shape)

def extract_genimage_frequency_features(image: Any, config: Mapping[str, Any] | None = None) -> np.ndarray:
    settings = _frequency_settings(config)
    image_size = int(settings.get("image_size", DEFAULT_FREQUENCY_IMAGE_SIZE))
    radial_bins = int(settings.get("radial_bins", DEFAULT_RADIAL_BINS))
    gray = image_to_grayscale_array(image, image_size=image_size)
    dct_spectrum = np.log1p(compute_dct_spectrum(gray))
    fft_spectrum = np.log1p(compute_fft_spectrum(gray))
    features = np.concatenate(
        [
            radial_average(dct_spectrum, radial_bins),
            radial_average(fft_spectrum, radial_bins),
            _energy_ratios(fft_spectrum),
            _block_dct_stats(gray),
        ]
    ).astype(FEATURE_DTYPE, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("GenImage frequency features must be finite")
    return features


def genimage_frequency_dim(config: Mapping[str, Any] | None = None) -> int:
    radial_bins = int(_frequency_settings(config).get("radial_bins", DEFAULT_RADIAL_BINS))
    return radial_bins * 2 + 4 + 10


def _energy_ratios(spectrum: np.ndarray) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    radius = _normalized_radius(values.shape)
    total = float(np.sum(values) + 1e-12)
    low = float(np.sum(values[radius < 0.25]) / total)
    mid = float(np.sum(values[(radius >= 0.25) & (radius < 0.60)]) / total)
    high = float(np.sum(values[radius >= 0.60]) / total)
    high_frequency_ratio = high / max(low + mid, 1e-12)
    return np.asarray([high_frequency_ratio, low, mid, high], dtype=FEATURE_DTYPE)


def _block_dct_stats(gray: np.ndarray, block_size: int = 8) -> np.ndarray:
    array = _validate_gray(gray)
    height = array.shape[0] - array.shape[0] % block_size
    width = array.shape[1] - array.shape[1] % block_size
    if height == 0 or width == 0:
        return np.zeros(10, dtype=FEATURE_DTYPE)
    values: list[float] = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            coeff = compute_dct_spectrum(array[y:y + block_size, x:x + block_size])
            high = coeff[block_size // 2 :, block_size // 2 :]
            values.append(float(np.mean(np.abs(high))))
    stats = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(stats, [0.1, 0.25, 0.5, 0.75, 0.9]) if stats.size else np.zeros(5)
    output = np.asarray([stats.mean(), stats.std(), stats.max(), stats.min(), *quantiles, stats.size], dtype=FEATURE_DTYPE)
    return output
