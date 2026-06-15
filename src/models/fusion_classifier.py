from __future__ import annotations

import torch
from torch import nn

from .fusion_mlp import FusionMLP, ResidualMLPBlock


class FusionClassifier(nn.Module):
    def __init__(self, clip_dim: int, freq_dim: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.freq_dim = int(freq_dim)
        self.input_dim = self.clip_dim + self.freq_dim
        self.classifier = FusionMLP(input_dim=self.input_dim, clip_dim=self.clip_dim, freq_dim=self.freq_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
