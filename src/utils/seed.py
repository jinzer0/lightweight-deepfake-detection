from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    seed_value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    try:
        import torch
    except ModuleNotFoundError:
        return

    _ = torch.random.manual_seed(seed_value)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
