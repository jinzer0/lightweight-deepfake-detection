from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

from PIL import Image

from .manifest import CLASS_TO_LABEL, OK_STATUS, validate_manifest_rows, write_manifest


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE = "local_real_fake"
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


def _select_subset(candidates: list[Candidate], seed: int, num_real: int | None, num_fake: int | None) -> list[Candidate]:
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
    return ["train"] * train_count + ["val"] * val_count + ["test"] * (count - train_count - val_count)


def _build_row(root: Path, candidate: Candidate, split: str, source: str) -> dict[str, object]:
    path = Path(candidate["path"])
    rel_path = candidate["rel_path"]
    class_name = candidate["class_name"]
    sha256 = _sha256(path)
    width, height = _image_size(path)
    stat = path.stat()
    sample_id = _sample_id(rel_path, sha256)
    return {
        "sample_id": sample_id,
        "base_sample_id": sample_id,
        "rel_path": rel_path,
        "root": root.as_posix(),
        "label": str(CLASS_TO_LABEL[class_name]),
        "class_name": class_name,
        "source": source,
        "source_split": candidate["source_split"],
        "split": split,
        "width": str(width),
        "height": str(height),
        "sha256": sha256,
        "file_size": str(stat.st_size),
        "mtime": str(stat.st_mtime_ns),
        "status": OK_STATUS,
    }


def _mark_duplicates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    output: list[dict[str, object]] = []
    for row in rows:
        marked = dict(row)
        sha = str(row["sha256"])
        if sha in seen:
            marked["status"] = "duplicate"
        seen.add(sha)
        output.append(marked)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _sample_id(rel_path: str, sha256: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", Path(rel_path).with_suffix("").as_posix()).strip("_").lower()
    return f"local_{stem}_{sha256[:12]}"
