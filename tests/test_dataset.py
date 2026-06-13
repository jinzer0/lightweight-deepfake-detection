from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ImageMetadataDataset
from src.data.make_dummy_dataset import make_dummy_dataset
from src.data.transforms import get_eval_transform, get_train_transform


def test_image_metadata_dataset_returns_tensor_label_and_metadata(tmp_path: Path) -> None:
    csv_path = _make_dummy_csv(tmp_path)
    dataset = ImageMetadataDataset(csv_path, split="train", transform=get_eval_transform(224))

    image, label, metadata = cast(tuple[torch.Tensor, int, dict[str, str]], dataset[0])

    assert isinstance(image, torch.Tensor)
    assert list(image.shape) == [3, 224, 224]
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert {"image_id", "filepath", "class_name", "dataset", "generator", "split"}.issubset(metadata)
    assert metadata["split"] == "train"


def test_image_metadata_dataset_dataloader_collates_batches(tmp_path: Path) -> None:
    csv_path = _make_dummy_csv(tmp_path)
    dataset = ImageMetadataDataset(csv_path, split="train", transform=get_eval_transform(224))

    images, labels, metadata = next(iter(DataLoader(dataset, batch_size=4)))

    assert list(images.shape) == [4, 3, 224, 224]
    assert list(labels.shape) == [4]
    assert labels.dtype == torch.long
    assert "image_id" in metadata
    assert len(metadata["image_id"]) == 4


def test_image_metadata_dataset_validates_split(tmp_path: Path) -> None:
    csv_path = _make_dummy_csv(tmp_path)

    with pytest.raises(ValueError, match="train.*val.*test"):
        ImageMetadataDataset(csv_path, split="dev")


def test_train_transform_resizes_and_normalizes(tmp_path: Path) -> None:
    csv_path = _make_dummy_csv(tmp_path)
    dataset = ImageMetadataDataset(csv_path, split="train", transform=get_train_transform(128), return_metadata=False)

    image, label = cast(tuple[torch.Tensor, int], dataset[0])

    assert list(image.shape) == [3, 128, 128]
    assert image.dtype == torch.float32
    assert isinstance(label, int)


def _make_dummy_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "metadata" / "dataset.csv"
    _ = make_dummy_dataset(num_real=6, num_fake=6, output_dir=tmp_path / "images", csv_path=csv_path, width=32, height=24)
    return csv_path
