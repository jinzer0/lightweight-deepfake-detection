from __future__ import annotations

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FusionMLP(nn.Module):
    def __init__(
        self,
        clip_dim: int | None = None,
        freq_dim: int | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        residual_blocks: int = 8,
        *,
        input_dim: int | None = None,
        num_blocks: int | None = None,
    ) -> None:
        super().__init__()
        block_count = residual_blocks if num_blocks is None else num_blocks
        if block_count != 8:
            raise ValueError('FusionMLP uses exactly 8 residual MLP blocks')
        if input_dim is None:
            if clip_dim is None or freq_dim is None:
                raise ValueError('clip_dim and freq_dim are required when input_dim is not provided')
            self.clip_dim = int(clip_dim)
            self.freq_dim = int(freq_dim)
            self.input_dim = self.clip_dim + self.freq_dim
        else:
            self.input_dim = int(input_dim)
            self.clip_dim = int(clip_dim or 0)
            self.freq_dim = int(freq_dim or 0)
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualMLPBlock(self.hidden_dim, dropout) for _ in range(8)])
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f'expected fusion features with shape (batch, {self.input_dim}), got {tuple(x.shape)}')
        return self.output(self.blocks(self.input(x))).squeeze(1)


ResidualBlock = ResidualMLPBlock
