from __future__ import annotations

import torch
from torch import nn


class ResNet50Baseline(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except Exception as exc:
            raise ImportError('torchvision is required for ResNet50Baseline') from exc
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        in_features = int(self.model.fc.in_features)
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f'expected RGB tensor batch (batch, 3, h, w), got {tuple(x.shape)}')
        return self.model(x).squeeze(1)
