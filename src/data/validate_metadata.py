from __future__ import annotations

import argparse

# pyright: reportAny=false, reportImplicitStringConcatenation=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
import csv
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

DATASET_COLUMNS = [
    "image_id",
    "filepath",
    "label",
    "class_name",
    "dataset",
    "generator",
    "split",
    "width",
    "height",
    "ext",
]
CLASS_TO_LABEL = {"real": 0, "fake": 1}
PROJECT_SPLITS = {"train", "val", "test"}


class MetadataValidationError(ValueError):
    pass


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_metadata(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=DATASET_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in DATASET_COLUMNS})


def validate_metadata(path: str | Path, strict: bool = True, print_counts: bool = False) -> list[str]:
    csv_path = Path(path)
    rows = read_metadata(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        header = next(csv.reader(file_obj), list[str]())
    errors = validate_metadata_rows(rows, header=header, strict=False)
    if print_counts:
        print_metadata_counts(rows)
    if errors and strict:
        raise MetadataValidationError("; ".join(errors))
    return errors


def validate_metadata_rows(
    rows: list[dict[str, str]],
    header: Sequence[str] | None = None,
    strict: bool = True,
    check_files: bool = True,
) -> list[str]:
    errors: list[str] = []
    if header is not None:
        missing_columns = [column for column in DATASET_COLUMNS if column not in header]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        if list(header) != DATASET_COLUMNS:
            errors.append(f"Invalid column order: expected {DATASET_COLUMNS}, got {list(header)}")
    elif not rows:
        errors.append("metadata has no rows")
    else:
        row_columns = list(rows[0].keys())
        missing_columns = [column for column in DATASET_COLUMNS if column not in row_columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        if row_columns != DATASET_COLUMNS:
            errors.append(f"Invalid column order: expected {DATASET_COLUMNS}, got {row_columns}")

    if not rows:
        errors.append("metadata has no rows")
        if errors and strict:
            raise MetadataValidationError("; ".join(errors))
        return errors

    _collect_duplicate_errors(rows, "image_id", errors)
    _collect_duplicate_errors(rows, "filepath", errors)

    missing_files: list[str] = []
    invalid_labels: set[str] = set()
    invalid_class_names: set[str] = set()
    invalid_splits: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        image_id = row.get("image_id", "")
        if not image_id:
            errors.append(f"row {row_number}: image_id is empty")

        filepath = row.get("filepath", "")
        if not filepath:
            errors.append(f"row {row_number}: filepath is empty")
        elif check_files and not Path(filepath).exists():
            missing_files.append(filepath)

        label_text = row.get("label", "")
        try:
            label = int(label_text)
        except ValueError:
            label = None
            invalid_labels.add(label_text)
        if label not in {0, 1}:
            invalid_labels.add(label_text)

        class_name = row.get("class_name", "")
        if class_name not in CLASS_TO_LABEL:
            invalid_class_names.add(class_name)
        elif label is not None and CLASS_TO_LABEL[class_name] != label:
            message = f"row {row_number} image_id={image_id}: label/class_name mismatch: "
            message += f"label={label_text}, class_name={class_name}"
            errors.append(message)

        split = row.get("split", "")
        if split not in PROJECT_SPLITS:
            invalid_splits.add(split)

        for dimension in ("width", "height"):
            value = row.get(dimension, "")
            try:
                if int(value) <= 0:
                    errors.append(f"row {row_number} image_id={image_id}: {dimension} must be positive, got {value}")
            except ValueError:
                errors.append(f"row {row_number} image_id={image_id}: {dimension} is not an integer: {value}")

        ext = row.get("ext", "")
        if not ext:
            errors.append(f"row {row_number} image_id={image_id}: ext is empty")

    if invalid_labels:
        errors.append(f"Invalid labels found: {sorted(invalid_labels)}")
    if invalid_class_names:
        errors.append(f"Invalid class_name found: {sorted(invalid_class_names)}")
    if invalid_splits:
        errors.append(f"Invalid splits found: {sorted(invalid_splits)}")
    if missing_files:
        errors.append(f"Missing files: {missing_files[:20]}")

    if errors and strict:
        raise MetadataValidationError("; ".join(errors))
    return errors


def print_metadata_counts(rows: list[dict[str, str]]) -> None:
    print("Split counts:")
    _print_counter(Counter(row.get("split", "") for row in rows))
    print("Split-by-label counts:")
    split_label_counts = Counter((row.get("split", ""), row.get("class_name", "")) for row in rows)
    for split in sorted({key[0] for key in split_label_counts}):
        for class_name in ("real", "fake"):
            print(f"  {split}/{class_name}: {split_label_counts[(split, class_name)]}")
    print("Generator counts:")
    _print_counter(Counter(row.get("generator", "") for row in rows))
    print("Dataset counts:")
    _print_counter(Counter(row.get("dataset", "") for row in rows))


def _collect_duplicate_errors(rows: list[dict[str, str]], column: str, errors: list[str]) -> None:
    counts = Counter(row.get(column, "") for row in rows)
    duplicates = sorted(value for value, count in counts.items() if value and count > 1)
    if duplicates:
        errors.append(f"Duplicated {column} found: {duplicates}")


def _print_counter(counter: Counter[str]) -> None:
    for key in sorted(counter):
        print(f"  {key}: {counter[key]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical dataset.csv metadata.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to dataset.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        errors = validate_metadata(args.csv, strict=True, print_counts=True)
    except MetadataValidationError as exc:
        print(f"Metadata validation failed: {exc}")
        raise SystemExit(1) from exc
    if errors:
        print("Metadata validation failed")
        raise SystemExit(1)
    print(f"Metadata validation passed: {args.csv}")


if __name__ == "__main__":
    main()
