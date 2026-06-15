from __future__ import annotations

import numpy as np

from src.eval.metrics import compute_binary_metrics


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float | str | int | list[list[int]] | None]:
    return compute_binary_metrics(labels, probs, threshold=threshold)
