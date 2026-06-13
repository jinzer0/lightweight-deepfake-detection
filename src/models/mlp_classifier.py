from __future__ import annotations

# pyright: reportAny=false, reportImplicitOverride=false, reportUnnecessaryIsInstance=false

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim: int = _positive_int(input_dim, "input_dim")
        self.hidden_dim: int = _positive_int(hidden_dim, "hidden_dim")
        self.dropout: float = _dropout_probability(dropout)
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(x)
        return logits.squeeze(1)


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _dropout_probability(value: float) -> float:
    dropout = float(value)
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {value}")
    return dropout
