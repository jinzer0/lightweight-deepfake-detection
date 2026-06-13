from __future__ import annotations

import hashlib
import random
import re
import struct
from pathlib import Path

from .manifest import CLASS_TO_LABEL, OK_STATUS, validate_manifest_rows, write_manifest


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE = "cifake_local"
SOURCE_SPLIT_ALIASES = {"train", "training", "val", "valid", "validation", "test", "testing"}
Candidate = dict[str, str]


def normalize_class_name(name: str) -> str | None:
    compact = re.sub(r"[^a-z0-9]+", "", name.lower())
    real_names = {"real", "reals", "true", "authentic", "original", "human"}
    fake_names = {"fake", "fakes", "false", "synthetic", "generated", "ai", "aigenerated"}
    if compact in real_names or compact.endswith("real"):
        return "real"
    if compact in fake_names or compact.endswith("fake"):
        return "fake"
    return None


def generate_manifest(
    data_root: str | Path,
    seed: int = 42,
    num_real: int | None = None,
    num_fake: int | None = None,
    source: str = SOURCE,
) -> list[dict[str, object]]:
    root = Path(data_root).resolve()
    candidates = _scan_candidates(root)
    selected = _select_subset(candidates, seed=seed, num_real=num_real, num_fake=num_fake)
    split_by_path = _assign_project_splits(selected, seed=seed)
    rows = [_build_row(root, candidate, split_by_path[candidate["rel_path"]], source) for candidate in selected]
    return _mark_duplicates(rows)


def generate_and_write_manifest(
    data_root: str | Path,
    output_manifest: str | Path,
    seed: int = 42,
    num_real: int | None = None,
    num_fake: int | None = None,
    strict: bool = True,
) -> list[str]:
    rows = generate_manifest(data_root, seed=seed, num_real=num_real, num_fake=num_fake)
    write_manifest(output_manifest, rows)
    return validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in rows], strict=strict)


def _scan_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_path = path.relative_to(root).as_posix()
        parts = Path(rel_path).parts
        class_name = None
        class_index = None
        for index, part in enumerate(parts[:-1]):
            normalized = normalize_class_name(part)
            if normalized is not None:
                class_name = normalized
                class_index = index
                break
        if class_name is None or class_index is None:
            continue
        source_split = ""
        if class_index > 0 and parts[class_index - 1].lower() in SOURCE_SPLIT_ALIASES:
            source_split = parts[class_index - 1].lower()
        candidates.append({"path": path.as_posix(), "rel_path": rel_path, "class_name": class_name, "source_split": source_split})
    return sorted(candidates, key=lambda candidate: str(candidate["rel_path"]))


def _select_subset(
    candidates: list[Candidate],
    seed: int,
    num_real: int | None,
    num_fake: int | None,
) -> list[Candidate]:
    limits: dict[str, int | None] = {"real": num_real, "fake": num_fake}
    randomizer = random.Random(seed)
    selected: list[Candidate] = []
    for class_name in ("real", "fake"):
        class_candidates = [candidate for candidate in candidates if candidate["class_name"] == class_name]
        shuffled = class_candidates[:]
        randomizer.shuffle(shuffled)
        limit = limits[class_name]
        selected.extend(shuffled if limit is None else shuffled[:limit])
    return sorted(selected, key=lambda candidate: candidate["rel_path"])


def _assign_project_splits(candidates: list[Candidate], seed: int) -> dict[str, str]:
    split_by_path: dict[str, str] = {}
    randomizer = random.Random(seed)
    for class_name in ("real", "fake"):
        class_candidates = [candidate for candidate in candidates if candidate["class_name"] == class_name]
        shuffled = class_candidates[:]
        randomizer.shuffle(shuffled)
        split_names = _balanced_split_names(len(shuffled))
        for candidate, split_name in zip(shuffled, split_names, strict=True):
            split_by_path[candidate["rel_path"]] = split_name
    return split_by_path


def _balanced_split_names(count: int) -> list[str]:
    train_count = int(count * 0.70)
    val_count = int(count * 0.15)
    if count >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = count - train_count - val_count
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def _build_row(root: Path, candidate: Candidate, split: str, source: str) -> dict[str, object]:
    path = Path(candidate["path"])
    rel_path = candidate["rel_path"]
    class_name = candidate["class_name"]
    sample_id = _sample_id(rel_path)
    width = ""
    height = ""
    sha256 = ""
    file_size: int | str = ""
    mtime: float | str = ""
    status: str = OK_STATUS
    try:
        stat = path.stat()
        file_size = stat.st_size
        mtime = stat.st_mtime
        sha256 = _sha256(path)
        width, height = _read_image_size(path)
    except FileNotFoundError:
        status = "missing"
    except Exception:
        status = "corrupt"
    return {
        "sample_id": sample_id,
        "base_sample_id": sample_id,
        "rel_path": rel_path,
        "root": root.as_posix(),
        "label": CLASS_TO_LABEL[class_name],
        "class_name": class_name,
        "source": source,
        "source_split": candidate.get("source_split", ""),
        "split": split,
        "width": width,
        "height": height,
        "sha256": sha256,
        "file_size": file_size,
        "mtime": mtime,
        "status": status,
    }


def _mark_duplicates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    first_by_hash: dict[str, dict[str, object]] = {}
    for row in rows:
        sha256 = str(row.get("sha256", ""))
        if not sha256:
            continue
        first = first_by_hash.setdefault(sha256, row)
        if first is not row:
            row["status"] = "duplicate"
            if first.get("status") == OK_STATUS:
                first["status"] = "duplicate"
    return rows


def _sample_id(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    return f"cifake_{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file_obj:
        header = file_obj.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR":
            return struct.unpack(">II", header[16:24])
        if header.startswith(b"\xff\xd8"):
            return _read_jpeg_size(path)
    raise OSError(f"unsupported or corrupt image: {path}")


def _read_jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as file_obj:
        _ = file_obj.read(2)
        while True:
            marker_start = file_obj.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = file_obj.read(1)
            while marker == b"\xff":
                marker = file_obj.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                segment = file_obj.read(7)
                height = int.from_bytes(segment[3:5], "big")
                width = int.from_bytes(segment[5:7], "big")
                return width, height
            length_bytes = file_obj.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big")
            _ = file_obj.seek(segment_length - 2, 1)
    raise OSError(f"unsupported or corrupt jpeg: {path}")
