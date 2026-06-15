from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAny=false

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.frequency_features import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RADIAL_BINS,
    compute_dct_spectrum,
    compute_fft_spectrum,
    image_to_grayscale_array,
    radial_average,
)

SUPPORTED_METHODS = ("dct", "fft")


def radial_spectrum_from_image(
    image: Any,
    config: Mapping[str, Any] | None = None,
    *,
    method: str | None = None,
    image_size: int | None = None,
    radial_bins: int | None = None,
    log_scale: bool | None = None,
    normalize_feature: bool | None = None,
) -> np.ndarray:
    settings = _frequency_settings(config)
    selected_method = str(method or settings.get("method", "dct")).lower()
    if selected_method not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unsupported frequency method '{selected_method}'. Supported methods: {supported}.")

    size = int(image_size if image_size is not None else settings.get("image_size", DEFAULT_IMAGE_SIZE))
    bins = int(radial_bins if radial_bins is not None else settings.get("radial_bins", DEFAULT_RADIAL_BINS))
    use_log_scale = bool(settings.get("log_scale", False) if log_scale is None else log_scale)
    use_normalize = bool(settings.get("normalize_feature", False) if normalize_feature is None else normalize_feature)

    gray = image_to_grayscale_array(image, image_size=size)
    spectrum = compute_fft_spectrum(gray) if selected_method == "fft" else compute_dct_spectrum(gray)
    values = radial_average(spectrum, bins=bins)
    if use_log_scale:
        values = np.log1p(values).astype(np.float32, copy=False)
    if use_normalize:
        norm = float(np.linalg.norm(values))
        if norm > 0.0:
            values = (values / norm).astype(np.float32, copy=False)
    return _validate_radial_values(values)


def save_radial_spectrum_plot(
    radial_spectrum: np.ndarray,
    output_path: str,
    *,
    method: str | None = None,
    log_scale: bool | None = None,
    normalize_feature: bool | None = None,
) -> str:
    values = _validate_radial_values(radial_spectrum)
    selected_method = method.upper() if method else "Frequency"

    transforms: list[str] = []
    if log_scale:
        transforms.append("log1p")
    if normalize_feature:
        transforms.append("L2-normalized")
    transform_text = f" ({', '.join(transforms)})" if transforms else ""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 3), dpi=100)
    axis.plot(np.arange(values.shape[0], dtype=np.int32), values)
    axis.set_title(f"{selected_method} radial spectrum{transform_text}")
    axis.set_xlabel("radial frequency bin")
    axis.set_ylabel("energy")
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)
    return str(path)


def _frequency_settings(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    frequency = config.get("frequency")
    return frequency if isinstance(frequency, Mapping) else config


def _validate_radial_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"radial spectrum feature must be 1D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("radial spectrum feature must contain only finite values")
    return values
