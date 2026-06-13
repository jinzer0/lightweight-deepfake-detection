from __future__ import annotations

# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportImplicitOverride=false

from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from src.utils.image_io import load_rgb_image


ALLOWED_SPLITS = ("train", "val", "test")
METADATA_COLUMNS = ("image_id", "filepath", "class_name", "dataset", "generator", "split")


class ImageMetadataDataset(Dataset[tuple[Any, int, dict[str, str]] | tuple[Any, int]]):
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform: Callable[[Any], Any] | None = None,
        return_metadata: bool = True,
    ) -> None:
        if split not in ALLOWED_SPLITS:
            allowed = ", ".join(ALLOWED_SPLITS)
            raise ValueError(f"Invalid split {split!r}; expected one of: {allowed}")

        self.csv_path: Path = Path(csv_path)
        self.split: str = split
        self.transform: Callable[[Image.Image], Any] | None = transform
        self.return_metadata: bool = return_metadata
        self.rows: list[dict[str, str]] = [row for row in _read_dataset_rows(self.csv_path) if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]] | tuple[Any, int]:
        row = self.rows[index]
        image = load_rgb_image(row["filepath"])
        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        if not self.return_metadata:
            return image, label

        metadata = {column: row[column] for column in METADATA_COLUMNS}
        return image, label, metadata


def _read_dataset_rows(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        return [
            {
                "image_id": row["image_id"],
                "filepath": row["filepath"],
                "label": row["label"],
                "class_name": row["class_name"],
                "dataset": row["dataset"],
                "generator": row["generator"],
                "split": row["split"],
            }
            for row in reader
        ]
