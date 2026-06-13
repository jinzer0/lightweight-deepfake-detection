from __future__ import annotations

# pyright: reportAny=false, reportImplicitOverride=false, reportUnnecessaryIsInstance=false

import torch
from torch import nn

from .mlp_classifier import MLPClassifier


class FusionClassifier(nn.Module):
    """MLP over concatenated CLIP and frequency features.

    The training and inference callers must concatenate features in the same
    order used here: ``input_dim = clip_dim + freq_dim``.
    """

    def __init__(self, clip_dim: int, freq_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.clip_dim: int = _positive_int(clip_dim, "clip_dim")
        self.freq_dim: int = _positive_int(freq_dim, "freq_dim")
        self.input_dim: int = self.clip_dim + self.freq_dim
        self.classifier: MLPClassifier = MLPClassifier(input_dim=self.input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.hidden_dim: int = self.classifier.hidden_dim
        self.dropout: float = self.classifier.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value
