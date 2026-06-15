from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = 'max') -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = self.best is None or (value > self.best if self.mode == 'max' else value < self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    import json
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
