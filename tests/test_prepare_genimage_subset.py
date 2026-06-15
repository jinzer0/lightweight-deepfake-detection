from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_genimage_subset import _drop_cross_split_duplicate_hashes
from src.data.manifest import validate_manifest_rows


def _manifest_row(root: Path, sample_id: str, split: str, digest: str) -> dict[str, object]:
    rel_path = f"{split}/real/real/{sample_id}.jpg"
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sample_id.encode("utf-8"))
    return {
        "sample_id": sample_id,
        "base_sample_id": sample_id,
        "rel_path": rel_path,
        "root": root.as_posix(),
        "label": "0",
        "class_name": "real",
        "source": "Tiny-GenImage:Real",
        "source_split": split,
        "split": split,
        "width": "8",
        "height": "8",
        "sha256": digest,
        "file_size": "1",
        "mtime": "1",
        "status": "ok",
    }


def test_drop_cross_split_duplicate_hashes_keeps_train_and_removes_leakage(tmp_path: Path) -> None:
    root = tmp_path / "images"
    train = _manifest_row(root, "train_sample", "train", "same_digest")
    test = _manifest_row(root, "test_sample", "test", "same_digest")
    val = _manifest_row(root, "val_sample", "val", "unique_digest")
    metadata_rows = [
        {"image_id": str(row["sample_id"]), "filepath": (root / str(row["rel_path"])).as_posix()}
        for row in [train, test, val]
    ]

    filtered_metadata, filtered_manifest, dropped = _drop_cross_split_duplicate_hashes(metadata_rows, [test, val, train])

    assert [row["sample_id"] for row in dropped] == ["test_sample"]
    assert {row["image_id"] for row in filtered_metadata} == {"train_sample", "val_sample"}
    assert {row["sample_id"] for row in filtered_manifest} == {"train_sample", "val_sample"}
    assert not (root / str(test["rel_path"])).exists()
    assert (root / str(train["rel_path"])).exists()
    validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in filtered_manifest], strict=True)
