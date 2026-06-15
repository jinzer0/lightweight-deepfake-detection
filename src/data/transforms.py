from __future__ import annotations

from typing import Any

from torchvision import transforms


def image_transform(image_size: int = 512, train: bool = False) -> Any:
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(ops)


def get_eval_transform(image_size: int = 512) -> Any:
    return image_transform(image_size=image_size, train=False)


def get_train_transform(image_size: int = 512) -> Any:
    return image_transform(image_size=image_size, train=True)
