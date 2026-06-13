from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportAny=false, reportExplicitAny=false

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.features.frequency_features import compute_dct_spectrum, compute_fft_spectrum, image_to_grayscale_array
from src.utils.image_io import DEFAULT_FREQUENCY_IMAGE_SIZE


SUPPORTED_METHODS = ("dct", "fft")


def save_spectrum_image(image: Any, output_path: str, method: str = "dct") -> str:
    selected_method = method.lower()
    if selected_method not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unsupported frequency method '{selected_method}'. Supported methods: {supported}.")

    gray = image_to_grayscale_array(image, image_size=DEFAULT_FREQUENCY_IMAGE_SIZE)
    spectrum = compute_dct_spectrum(gray) if selected_method == "dct" else compute_fft_spectrum(gray)
    log_spectrum = np.log1p(spectrum)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(4, 4), dpi=100)
    axis.imshow(log_spectrum, cmap="gray")
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(path, format="png")
    plt.close(figure)
    return str(path)
