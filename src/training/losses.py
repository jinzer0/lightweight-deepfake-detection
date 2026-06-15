from __future__ import annotations

import torch


def binary_pos_weight(labels) -> torch.Tensor | None:
    y = torch.as_tensor(labels).float()
    pos = y.sum()
    neg = y.numel() - pos
    if pos <= 0:
        return None
    return torch.tensor([float(neg / pos)])
