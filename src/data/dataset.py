from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

ALLOWED_SPLITS = ("train", "val", "test")
METADATA_COLUMNS = ("image_id", "filepath", "class_name", "dataset", "generator", "source_split", "split")


class GenImageDataset(Dataset[tuple[Any, int, dict[str, str]] | tuple[Any, int]]):
    """PyTorch dataset backed by a GenImage manifest CSV."""

    def __init__(self, manifest_csv: str | Path, split: str, transform: Callable[[Image.Image], Any] | None = None) -> None:
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split {split!r}; expected one of {ALLOWED_SPLITS}")
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.transform = transform
        self.rows = [row for row in read_rows(self.manifest_csv) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r} in {self.manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]] | tuple[Any, int]:
        row = self.rows[index]
        image_path = row.get("filepath") or row.get("path")
        if image_path is None:
            raise ValueError("row is missing filepath/path")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), dict(row)


class ImageMetadataDataset(GenImageDataset):
    """Backward-compatible dataset for both dataset.csv and GenImage manifest CSVs."""

    def __init__(self, csv_path: str | Path, split: str, transform: Callable[[Any], Any] | None = None, return_metadata: bool = True) -> None:
        self.return_metadata = return_metadata
        super().__init__(csv_path, split, transform)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]] | tuple[Any, int]:
        item = super().__getitem__(index)
        if len(item) != 3:
            raise ValueError("GenImageDataset must return image, label, metadata")
        image, label, metadata = item
        if self.return_metadata:
            return image, label, metadata
        return image, label


def read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    normalized: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        path = item.get("path") or item.get("filepath") or item.get("rel_path")
        if path is None:
            raise ValueError("manifest must contain path, filepath, or rel_path")
        item.setdefault("path", path)
        item.setdefault("filepath", path)
        item.setdefault("label", "1" if item.get("class_name") == "fake" else "0")
        item.setdefault("class_name", "fake" if str(item["label"]) == "1" else "real")
        item.setdefault("dataset", item.get("source", "GenImage"))
        item.setdefault("generator", item.get("dataset", "unknown"))
        item.setdefault("source_split", "unknown")
        item.setdefault("split", "train")
        item.setdefault("image_id", item.get("sample_id", f"row_{idx:06d}"))
        item.setdefault("width", "")
        item.setdefault("height", "")
        item.setdefault("ext", Path(path).suffix.lower().lstrip("."))
        normalized.append(item)
    return normalized
