from __future__ import annotations

import pytest
import torch

from src.models.clip_classifier import ClipClassifier
from src.models.frequency_classifier import FrequencyClassifier
from src.models.fusion_mlp import FusionMLP


def test_clip_classifier_shape_assertions() -> None:
    model = ClipClassifier(input_dim=8, hidden_dim=4, dropout=0.0)
    assert list(model(torch.randn(2, 8)).shape) == [2]
    with pytest.raises(ValueError):
        model(torch.randn(2, 7))


def test_frequency_classifier_shape_assertions() -> None:
    model = FrequencyClassifier(input_dim=6, hidden_dim=4, dropout=0.0)
    assert list(model(torch.randn(3, 6)).shape) == [3]
    with pytest.raises(ValueError):
        model(torch.randn(3, 5))


def test_fusion_mlp_has_eight_residual_blocks_and_shape_assertion() -> None:
    model = FusionMLP(clip_dim=4, freq_dim=3, hidden_dim=8, dropout=0.0)
    assert len(model.blocks) == 8
    assert list(model(torch.randn(5, 7)).shape) == [5]
    with pytest.raises(ValueError):
        model(torch.randn(5, 6))
