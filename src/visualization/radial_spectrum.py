from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAny=false

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_radial_spectrum_plot(feature: np.ndarray, output_path: str) -> str:
    values = np.asarray(feature, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"radial spectrum feature must be 1D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("radial spectrum feature must contain only finite values")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 3), dpi=100)
    axis.plot(np.arange(values.shape[0], dtype=np.int32), values)
    axis.set_xlabel("frequency bin")
    axis.set_ylabel("normalized energy")
    figure.tight_layout()
    figure.savefig(path, format="png")
    plt.close(figure)
    return str(path)
