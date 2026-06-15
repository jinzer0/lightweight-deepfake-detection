from __future__ import annotations

import argparse

import csv
import random
from pathlib import Path

from .validate_metadata import DATASET_COLUMNS, PROJECT_SPLITS, validate_metadata_rows, write_metadata


def assign_splits(
    rows: list[dict[str, str]],
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> list[dict[str, str]]:
    output_rows = [dict(row) for row in rows]
    randomizer = random.Random(seed)
    for label in sorted({row.get("label", "") for row in output_rows}):
        label_rows = [row for row in output_rows if row.get("label", "") == label]
        randomizer.shuffle(label_rows)
        split_names = _split_names(len(label_rows), train_ratio=train_ratio, val_ratio=val_ratio)
        for row, split in zip(label_rows, split_names, strict=True):
            row["split"] = split
    return sorted(output_rows, key=lambda row: row.get("image_id", ""))


def read_unsplit_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = []
        for row in reader:
            normalized = {column: row.get(column, "") for column in DATASET_COLUMNS}
            rows.append(normalized)
        return rows


def _split_names(count: int, train_ratio: float, val_ratio: float) -> list[str]:
    if count == 0:
        return []
    train_count = int(count * train_ratio)
    val_count = int(count * val_ratio)
    if count >= len(PROJECT_SPLITS):
        train_count = max(1, train_count)
        val_count = max(1, val_count)
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = count - train_count - val_count
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic stratified train/val/test splits for metadata CSV rows.")
    parser.add_argument("--input_csv", type=Path, required=True, help="Input CSV with empty or absent split values")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output canonical dataset.csv path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = read_unsplit_metadata(args.input_csv)
    split_rows = assign_splits(rows, seed=args.seed)
    errors = validate_metadata_rows(split_rows, header=DATASET_COLUMNS, strict=False, check_files=True)
    if errors:
        message = "; ".join(errors)
        print(f"Split metadata validation failed: {message}")
        raise SystemExit(1)
    write_metadata(args.output_csv, split_rows)
    print(f"Wrote split metadata: {args.output_csv}")


if __name__ == "__main__":
    main()
