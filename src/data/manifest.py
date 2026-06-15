from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "2"
GENIMAGE_MANIFEST_COLUMNS = ["path", "label", "class_name", "generator", "source_split", "split"]
MANIFEST_COLUMNS = [
    "sample_id", "base_sample_id", "rel_path", "root", "label", "class_name",
    "source", "source_split", "split", "width", "height", "sha256",
    "file_size", "mtime", "status",
]
CLASS_TO_LABEL = {"real": 0, "fake": 1}
LABEL_TO_CLASS = {0: "real", 1: "fake"}
PROJECT_SPLITS = {"train", "val", "test"}
OK_STATUS = "ok"
REAL_KEYWORDS = ("nature", "real", "natural")
FAKE_KEYWORDS = ("ai", "fake", "generated", "synth", "synthetic")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_GENERATORS = {"stable_diffusion", "midjourney", "biggan", "glide", "adm", "wukong", "vqdm"}


class ManifestValidationError(ValueError):
    pass


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> None:
    row_list = list(rows)
    fieldnames = columns or (list(row_list[0].keys()) if row_list else GENIMAGE_MANIFEST_COLUMNS)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def validate_manifest_rows(rows: list[dict[str, str]], strict: bool = True) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("manifest has no rows")
        if strict:
            raise ManifestValidationError("; ".join(errors))
        return errors
    required = MANIFEST_COLUMNS if "rel_path" in rows[0] else GENIMAGE_MANIFEST_COLUMNS
    missing_columns = [column for column in required if column not in rows[0]]
    if missing_columns:
        errors.append(f"manifest missing required columns: {', '.join(missing_columns)}")
    seen_paths: set[str] = set()
    split_paths: dict[str, str] = {}
    seen_ids: set[str] = set()
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    for row_number, row in enumerate(rows, start=2):
        row_path = row.get("path") or row.get("rel_path") or row.get("sample_id") or ""
        if not row_path:
            errors.append(f"row {row_number}: path is empty")
        if row_path in seen_paths:
            errors.append(f"row {row_number}: duplicate path {row_path}")
        seen_paths.add(row_path)
        sample_id = row.get("sample_id", "")
        if sample_id:
            if sample_id in seen_ids:
                errors.append(f"row {row_number}: duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
        split = row.get("split", "")
        if split not in PROJECT_SPLITS:
            errors.append(f"row {row_number}: invalid split {split}")
        existing = split_paths.get(row_path)
        if existing is not None and existing != split:
            errors.append(f"row {row_number}: path leakage across splits: {row_path}")
        split_paths[row_path] = split
        label_text = row.get("label", "")
        try:
            label = int(label_text)
        except ValueError:
            errors.append(f"row {row_number}: label is not an integer: {label_text}")
            label = -1
        if label not in LABEL_TO_CLASS:
            errors.append(f"row {row_number}: label must be 0 or 1, got {label_text}")
        class_name = row.get("class_name", "")
        if class_name in CLASS_TO_LABEL and CLASS_TO_LABEL[class_name] != label:
            errors.append(f"row {row_number}: label/class_name mismatch")
        digest = row.get("sha256", "")
        if digest:
            hash_to_splits[digest].add(split)
    for digest, splits in hash_to_splits.items():
        if len(splits) > 1:
            errors.append(f"duplicate sha256 leakage across splits: {digest}")
    if errors and strict:
        raise ManifestValidationError("; ".join(errors))
    return errors


def infer_label(path: Path) -> tuple[int, str] | None:
    tokens = {part.lower().replace("-", "_") for part in path.parts}
    stem_tokens = set(path.stem.lower().replace("-", "_").split("_"))
    all_tokens = tokens | stem_tokens
    real = bool(all_tokens.intersection(REAL_KEYWORDS))
    fake = bool(all_tokens.intersection(FAKE_KEYWORDS))
    if real and not fake:
        return 0, "real"
    if fake and not real:
        return 1, "fake"
    if "real" in all_tokens:
        return 0, "real"
    if fake:
        return 1, "fake"
    return None


def infer_generator(path: Path, data_root: Path) -> str:
    rel_parts = path.relative_to(data_root).parts
    lowered = [part.lower() for part in rel_parts[:-1]]
    for part in lowered:
        normalized = part.replace("-", "_")
        if normalized in KNOWN_GENERATORS:
            return normalized
    return lowered[0].replace("-", "_") if lowered else "unknown"


def infer_source_split(path: Path) -> str:
    for part in path.parts:
        lower = part.lower()
        if lower in PROJECT_SPLITS or lower in {"valid", "validation"}:
            return "val" if lower in {"valid", "validation"} else lower
    return "unknown"


def scan_genimage(data_root: str | Path, max_samples_per_generator: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"GenImage data_root does not exist: {root}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()):
        inferred = infer_label(image_path)
        if inferred is None:
            continue
        label, class_name = inferred
        generator = infer_generator(image_path, root)
        grouped[generator].append({
            "path": str(image_path), "label": label, "class_name": class_name,
            "generator": generator, "source_split": infer_source_split(image_path), "split": "",
        })
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for generator, items in sorted(grouped.items()):
        selected = list(items)
        if max_samples_per_generator is not None and len(selected) > max_samples_per_generator:
            selected = sorted(rng.sample(selected, int(max_samples_per_generator)), key=lambda row: row["path"])
        rows.extend(selected)
    if not rows:
        raise ValueError("No labeled image files found. Expected path keywords for real/nature or ai/fake/generated/synth/synthetic.")
    return rows


def assign_random_split(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(dict(row))
    output: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = max(1, int(round(n * 0.80))) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(round(n * 0.10))) if n >= 3 else (1 if n >= 2 else 0)
        if n_train + n_val >= n and n >= 3:
            n_train = n - 2
            n_val = 1
        for idx, row in enumerate(label_rows):
            row["split"] = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
            output.append(row)
    return sorted(output, key=lambda row: row["path"])


def assign_generator_holdout_split(rows: list[dict[str, Any]], holdout_generators: set[str], seed: int = 42) -> list[dict[str, Any]]:
    if not holdout_generators:
        raise ValueError("generator_holdout requires --holdout_generators")
    normalized_holdout = {item.strip().lower().replace("-", "_") for item in holdout_generators if item.strip()}
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        if str(copy["generator"]).lower().replace("-", "_") in normalized_holdout:
            copy["split"] = "test"
            test_rows.append(copy)
        else:
            train_rows.append(copy)
    if not test_rows:
        raise ValueError(f"No rows matched holdout generators: {sorted(normalized_holdout)}")
    val_split = assign_random_split(train_rows, seed=seed)
    output = []
    for row in val_split:
        if row["split"] == "test":
            row["split"] = "val"
        output.append(row)
    output.extend(test_rows)
    train_gens = {row["generator"] for row in output if row["split"] in {"train", "val"}}
    test_gens = {row["generator"] for row in output if row["split"] == "test"}
    overlap = train_gens & test_gens
    if overlap:
        raise ValueError(f"train/test generator leakage: {sorted(overlap)}")
    return sorted(output, key=lambda row: row["path"])


def build_manifest_rows(data_root: str | Path, split_mode: str = "random", holdout_generators: str = "", max_samples_per_generator: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    rows = scan_genimage(data_root, max_samples_per_generator=max_samples_per_generator, seed=seed)
    if split_mode == "random":
        return assign_random_split(rows, seed=seed)
    if split_mode == "generator_holdout":
        return assign_generator_holdout_split(rows, set(holdout_generators.split(",")), seed=seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_hashes(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        hashes[sha256_file(row["path"])].append(row["path"])
    return {digest: paths for digest, paths in hashes.items() if len(paths) > 1}


def validate_manifest(path: str | Path, strict: bool = True) -> list[str]:
    return validate_manifest_rows(read_manifest(path), strict=strict)
