from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_synthbuster_ood import prepare_synthbuster_ood
from src.data.manifest import read_manifest, validate_manifest_rows
from src.data.validate_metadata import DATASET_COLUMNS, read_metadata, validate_metadata_rows


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (17, 19), color=color).save(path, format="PNG")


def _make_synthbuster_zip(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write_png(source / "synthbuster" / "glide" / "a.png", (10, 20, 30))
    _write_png(source / "synthbuster" / "dalle2" / "b.png", (40, 50, 60))
    (source / "synthbuster" / "prompts.csv").write_text("filename,prompt\na.png,test\n", encoding="utf-8")
    zip_path = tmp_path / "synthbuster.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    return zip_path


def test_prepare_synthbuster_ood_writes_test_only_metadata_and_manifest(tmp_path: Path) -> None:
    zip_path = _make_synthbuster_zip(tmp_path)
    output_dir = tmp_path / "data" / "synthbuster"
    metadata_csv = tmp_path / "data" / "metadata" / "synthbuster_ood_dataset.csv"
    manifest_csv = tmp_path / "outputs" / "synthbuster_ood" / "manifest.csv"

    rows = prepare_synthbuster_ood(
        zip_path=zip_path,
        output_dir=output_dir,
        metadata_csv=metadata_csv,
        manifest_path=manifest_csv,
        clean=False,
        copy_docs=True,
    )

    assert len(rows) == 2
    assert metadata_csv.is_file()
    assert manifest_csv.is_file()
    assert (output_dir / "test" / "glide" / "fake" / "a.png").is_file()
    assert (output_dir / "test" / "dalle2" / "fake" / "b.png").is_file()
    assert (output_dir / "_docs" / "prompts.csv").is_file()

    with metadata_csv.open("r", newline="", encoding="utf-8") as file_obj:
        assert next(csv.reader(file_obj)) == DATASET_COLUMNS
    metadata_rows = read_metadata(metadata_csv)
    assert {row["dataset"] for row in metadata_rows} == {"Synthbuster-OOD"}
    assert {row["class_name"] for row in metadata_rows} == {"fake"}
    assert {row["label"] for row in metadata_rows} == {"1"}
    assert {row["split"] for row in metadata_rows} == {"test"}
    assert validate_metadata_rows(metadata_rows, header=DATASET_COLUMNS, strict=False, check_files=True) == []

    manifest_rows = read_manifest(manifest_csv)
    assert {row["root"] for row in manifest_rows} == {output_dir.resolve().as_posix()}
    assert {row["rel_path"] for row in manifest_rows} == {"test/dalle2/fake/b.png", "test/glide/fake/a.png"}
    assert {row["source_split"] for row in manifest_rows} == {"ood"}
    assert validate_manifest_rows(manifest_rows, strict=False) == []
