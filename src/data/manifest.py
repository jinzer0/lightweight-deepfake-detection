from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path


MANIFEST_VERSION = "1"
MANIFEST_COLUMNS = [
    "sample_id",
    "base_sample_id",
    "rel_path",
    "root",
    "label",
    "class_name",
    "source",
    "source_split",
    "split",
    "width",
    "height",
    "sha256",
    "file_size",
    "mtime",
    "status",
]

CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_CLASS = {0: "real", 1: "fake"}
PROJECT_SPLITS = {"train", "val", "test"}
OK_STATUS = "ok"


class ManifestValidationError(ValueError):
    pass


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_manifest(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def validate_manifest_rows(rows: list[dict[str, str]], strict: bool = True) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("manifest has no rows")
        if strict:
            raise ManifestValidationError("; ".join(errors))
        return errors

    missing_columns = [column for column in MANIFEST_COLUMNS if column not in rows[0]]
    if missing_columns:
        errors.append(f"manifest missing required columns: {', '.join(missing_columns)}")

    seen_sample_ids: set[str] = set()
    hash_to_splits: dict[str, set[str]] = {}
    hash_to_paths: dict[str, set[str]] = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "")
        if not sample_id:
            errors.append(f"row {row_number}: sample_id is empty")
        elif sample_id in seen_sample_ids:
            errors.append(f"row {row_number}: duplicate sample_id {sample_id}")
        seen_sample_ids.add(sample_id)

        class_name = row.get("class_name", "")
        label_text = row.get("label", "")
        try:
            label = int(label_text)
        except ValueError:
            errors.append(f"row {row_number}: label is not an integer: {label_text}")
            label = None
        if class_name not in CLASS_TO_LABEL:
            errors.append(f"row {row_number}: class_name must be real or fake, got {class_name}")
        elif label != CLASS_TO_LABEL[class_name]:
            errors.append(
                f"row {row_number}: label polarity mismatch for {class_name}: expected {CLASS_TO_LABEL[class_name]}, got {label_text}"
            )

        split = row.get("split", "")
        if split not in PROJECT_SPLITS:
            errors.append(f"row {row_number}: split must be train, val, or test, got {split}")

        status = row.get("status", "")
        if strict and status != OK_STATUS:
            errors.append(f"row {row_number}: status is {status}, expected ok")

        sha256 = row.get("sha256", "")
        if sha256:
            hash_to_splits.setdefault(sha256, set()).add(split)
            hash_to_paths.setdefault(sha256, set()).add(row.get("rel_path", ""))

    for sha256, splits in sorted(hash_to_splits.items()):
        if len(splits) > 1:
            paths = sorted(hash_to_paths.get(sha256, set()))
            message = f"duplicate sha256 leakage across project splits {sorted(splits)} "
            message += f"for hash {sha256}: {', '.join(paths)}"
            errors.append(message)

    if errors and strict:
        raise ManifestValidationError("; ".join(errors))
    return errors


def validate_manifest(path: str | Path, strict: bool = True) -> list[str]:
    return validate_manifest_rows(read_manifest(path), strict=strict)
