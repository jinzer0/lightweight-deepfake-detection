from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from pathlib import Path

import numpy as np

from src.features.frequency import extract_frequency_features


def test_frequency_features_are_finite_with_expected_dimension(tiny_png: Path) -> None:
    features = extract_frequency_features(tiny_png)
    assert features.dtype == np.float32
    assert features.ndim == 1
    assert 100 <= features.shape[0] <= 250
    assert features.shape[0] == 220
    assert np.isfinite(features).all()
