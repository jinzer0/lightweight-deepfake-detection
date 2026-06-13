from __future__ import annotations

import csv

# pyright: reportArgumentType=false, reportImplicitStringConcatenation=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
from pathlib import Path

import pytest

from src.data.make_dummy_dataset import make_dummy_dataset
from src.data.make_split import assign_splits, read_unsplit_metadata
from src.data.validate_metadata import (
    DATASET_COLUMNS,
    MetadataValidationError,
    validate_metadata,
    validate_metadata_rows,
)


def test_dummy_dataset_writes_canonical_metadata_and_validates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "raw" / "dummy"
    csv_path = tmp_path / "metadata" / "dataset.csv"

    rows = make_dummy_dataset(num_real=6, num_fake=6, output_dir=output_dir, csv_path=csv_path, width=17, height=19, seed=123)
    repeated_rows = make_dummy_dataset(num_real=6, num_fake=6, output_dir=output_dir, csv_path=csv_path, width=17, height=19, seed=123)

    assert rows == repeated_rows
    assert csv_path.exists()
    assert len(rows) == 12
    assert list(rows[0].keys()) == DATASET_COLUMNS
    assert {row["class_name"] for row in rows} == {"real", "fake"}
    assert {int(row["label"]) for row in rows} == {0, 1}
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(Path(row["filepath"]).exists() for row in rows)
    assert all(row["width"] == "17" and row["height"] == "19" and row["ext"] == "png" for row in rows)

    validate_metadata(csv_path, strict=True, print_counts=True)
    captured = capsys.readouterr().out
    assert "Split counts:" in captured
    assert "Split-by-label counts:" in captured
    assert "Generator counts:" in captured
    assert "Dataset counts:" in captured


def test_metadata_rejects_label_class_mismatch_duplicates_bad_split_and_missing_file(tmp_path: Path) -> None:
    existing_image = tmp_path / "real.png"
    existing_image.write_bytes(b"not used by validator")
    missing_image = tmp_path / "missing.png"
    rows = [
        {
            "image_id": "sample_1",
            "filepath": existing_image.as_posix(),
            "label": "0",
            "class_name": "real",
            "dataset": "DUMMY",
            "generator": "real_dummy",
            "split": "train",
            "width": "10",
            "height": "12",
            "ext": "png",
        },
        {
            "image_id": "sample_1",
            "filepath": existing_image.as_posix(),
            "label": "0",
            "class_name": "fake",
            "dataset": "DUMMY",
            "generator": "dummy_generator",
            "split": "holdout",
            "width": "10",
            "height": "12",
            "ext": "png",
        },
        {
            "image_id": "sample_3",
            "filepath": missing_image.as_posix(),
            "label": "2",
            "class_name": "fake",
            "dataset": "DUMMY",
            "generator": "dummy_generator",
            "split": "test",
            "width": "10",
            "height": "12",
            "ext": "png",
        },
    ]

    with pytest.raises(MetadataValidationError) as exc_info:
        validate_metadata_rows(rows, header=DATASET_COLUMNS, strict=True)

    message = str(exc_info.value)
    assert "Duplicated image_id found" in message
    assert "Duplicated filepath found" in message
    assert "label/class_name mismatch" in message
    assert "image_id=sample_1" in message
    assert "Invalid labels found" in message
    assert "Invalid splits found" in message
    assert "Missing files" in message


def test_metadata_rejects_missing_or_reordered_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    bad_csv = "filepath,image_id,label,class_name,dataset,generator,split,width,height,ext\n"
    bad_csv += "x.png,sample,0,real,DUMMY,real_dummy,train,10,10,png\n"
    csv_path.write_text(bad_csv, encoding="utf-8")

    with pytest.raises(MetadataValidationError, match="Invalid column order"):
        validate_metadata(csv_path, strict=True)

    missing_column_path = tmp_path / "missing.csv"
    missing_csv = "image_id,filepath,label,class_name,dataset,split,width,height,ext\n"
    missing_csv += "sample,x.png,0,real,DUMMY,train,10,10,png\n"
    missing_column_path.write_text(missing_csv, encoding="utf-8")
    with pytest.raises(MetadataValidationError, match="Missing required columns"):
        validate_metadata(missing_column_path, strict=True)


def test_make_split_supports_unsplit_csv_and_is_deterministic(tmp_path: Path) -> None:
    paths = []
    for index in range(10):
        path = tmp_path / f"image_{index}.png"
        path.write_bytes(b"placeholder")
        paths.append(path)

    input_csv = tmp_path / "unsplit.csv"
    columns_without_split = [column for column in DATASET_COLUMNS if column != "split"]
    with input_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns_without_split)
        writer.writeheader()
        for index, path in enumerate(paths):
            class_name = "real" if index < 5 else "fake"
            writer.writerow(
                {
                    "image_id": f"sample_{index}",
                    "filepath": path.as_posix(),
                    "label": "0" if class_name == "real" else "1",
                    "class_name": class_name,
                    "dataset": "DUMMY",
                    "generator": "real_dummy" if class_name == "real" else "dummy_generator",
                    "width": "10",
                    "height": "10",
                    "ext": "png",
                }
            )

    rows = read_unsplit_metadata(input_csv)
    split_rows = assign_splits(rows, seed=7)
    repeated_rows = assign_splits(rows, seed=7)

    assert split_rows == repeated_rows
    assert list(split_rows[0].keys()) == DATASET_COLUMNS
    assert {row["split"] for row in split_rows} == {"train", "val", "test"}
    validate_metadata_rows(split_rows, header=DATASET_COLUMNS, strict=True)


def test_dataset_csv_example_uses_canonical_header() -> None:
    example = Path("data/metadata/dataset.csv.example")
    with example.open("r", newline="", encoding="utf-8") as file_obj:
        header = next(csv.reader(file_obj))
    assert header == DATASET_COLUMNS
