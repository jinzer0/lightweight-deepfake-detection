from __future__ import annotations

import torch
from torch import nn

from src.models.mlp_classifier import MLPClassifier


class FrequencyClassifier(nn.Module):
    def __init__(self, input_dim: int = 154, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.classifier = MLPClassifier(self.input_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'expected frequency features with shape (batch, {self.input_dim}), got {tuple(x.shape)}')
        return self.classifier(x)
