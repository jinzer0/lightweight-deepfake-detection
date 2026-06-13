from __future__ import annotations

# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_train_transform(image_size: int) -> Callable[[Any], torch.Tensor]:
    def transform(image: Any) -> torch.Tensor:
        resized = _resize_rgb(image, image_size)
        if random.random() < 0.5:
            resized = ImageOps.mirror(resized)
        return _normalize(_to_tensor(resized))

    return transform


def get_eval_transform(image_size: int) -> Callable[[Any], torch.Tensor]:
    def transform(image: Any) -> torch.Tensor:
        return _normalize(_to_tensor(_resize_rgb(image, image_size)))

    return transform


def _resize_rgb(image: Any, image_size: int) -> Image.Image:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    return rgb_image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std
