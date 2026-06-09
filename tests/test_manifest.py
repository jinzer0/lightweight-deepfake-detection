from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false

from pathlib import Path

import pytest

from src.data.cifake import generate_manifest
from src.data.manifest import CLASS_TO_LABEL, MANIFEST_COLUMNS, ManifestValidationError, validate_manifest_rows, write_manifest, read_manifest


def test_manifest_schema_label_polarity_and_deterministic_order(synthetic_cifake_root: Path, tmp_path: Path) -> None:
    rows = generate_manifest(synthetic_cifake_root, seed=123)
    repeated_rows = generate_manifest(synthetic_cifake_root, seed=123)
    assert rows == repeated_rows
    assert list(rows[0].keys()) == MANIFEST_COLUMNS
    assert [row["rel_path"] for row in rows] == sorted(row["rel_path"] for row in rows)
    assert {row["class_name"] for row in rows} == {"real", "fake"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(int(row["label"]) == CLASS_TO_LABEL[str(row["class_name"])] for row in rows)
    validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in rows], strict=True)

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)
    read_rows = read_manifest(manifest_path)
    assert read_rows[0].keys() == rows[0].keys()
    validate_manifest_rows(read_rows, strict=True)


def test_manifest_rejects_duplicate_sample_and_split_leakage(synthetic_cifake_root: Path) -> None:
    rows = [{key: str(value) for key, value in row.items()} for row in generate_manifest(synthetic_cifake_root, seed=9)]
    duplicate_id_rows = [dict(row) for row in rows]
    duplicate_id_rows[1]["sample_id"] = duplicate_id_rows[0]["sample_id"]
    with pytest.raises(ManifestValidationError, match="duplicate sample_id"):
        validate_manifest_rows(duplicate_id_rows, strict=True)

    leakage_rows = [dict(row) for row in rows]
    leakage_rows[1]["sha256"] = leakage_rows[0]["sha256"]
    leakage_rows[1]["split"] = "test" if leakage_rows[0]["split"] != "test" else "train"
    with pytest.raises(ManifestValidationError, match="duplicate sha256 leakage"):
        validate_manifest_rows(leakage_rows, strict=True)
