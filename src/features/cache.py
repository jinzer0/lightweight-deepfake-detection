from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch



SCHEMA_VERSION = "feature-cache-v1"
FEATURE_TYPES = {"frequency", "clip", "fusion"}


class FeatureCacheError(ValueError):
    pass


class TorchDependencyError(RuntimeError):
    pass


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("path") or row.get("filepath") or row.get("rel_path"))


def _row_path(row: Mapping[str, Any]) -> str:
    return str(row.get("path") or row.get("filepath") or row.get("rel_path"))


def hash_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    import hashlib, json
    payload = [{"id": _row_id(row), "path": _row_path(row), "label": str(row.get("label", "")), "split": str(row.get("split", ""))} for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def hash_manifest_file(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_feature_config(config: Mapping[str, Any]) -> str:
    import hashlib, json
    return hashlib.sha256(json.dumps(dict(config), sort_keys=True).encode("utf-8")).hexdigest()


def build_metadata(feature_dim: int, dtype: str, normalization: str, seed: int, extra: Mapping[str, Any] | None = None, created_at: str | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone
    return {"feature_dim": int(feature_dim), "dtype": dtype, "normalization": normalization, "seed": int(seed), "extra": dict(extra or {}), "created_at": created_at or datetime.now(timezone.utc).isoformat()}


def create_feature_cache(manifest_rows: Sequence[Mapping[str, Any]], feature_type: str, feature_config: Mapping[str, Any], features: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    if feature_type not in FEATURE_TYPES:
        raise FeatureCacheError(f"unsupported feature_type {feature_type}")
    rows = list(manifest_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_type": feature_type,
        "feature_config": dict(feature_config),
        "feature_config_hash": hash_feature_config(feature_config),
        "manifest_hash": hash_manifest_rows(rows),
        "sample_ids": [_row_id(row) for row in rows],
        "paths": [_row_path(row) for row in rows],
        "labels": [int(row.get("label", 0)) for row in rows],
        "splits": [str(row.get("split", "")) for row in rows],
        "features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
        "metadata": dict(metadata),
    }


def validate_feature_cache(cache: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]], expected_feature_config: Mapping[str, Any] | None = None) -> None:
    rows = list(manifest_rows)
    if cache.get("manifest_hash") != hash_manifest_rows(rows):
        raise FeatureCacheError("manifest_hash is stale")
    sample_ids = list(cache.get("sample_ids", []))
    if len(sample_ids) != len(set(sample_ids)):
        raise FeatureCacheError("sample_ids contains duplicates")
    expected_sample_ids = [_row_id(row) for row in rows]
    if sample_ids != expected_sample_ids:
        raise FeatureCacheError("sample_ids order or values do not match manifest")
    paths = list(cache.get("paths", []))
    expected_paths = [_row_path(row) for row in rows]
    if len(paths) != len(expected_paths):
        raise FeatureCacheError("paths length does not match manifest")
    if paths != expected_paths:
        raise FeatureCacheError("paths order or values do not match manifest")
    raw_features = cache.get("features")
    features = raw_features.detach().cpu().numpy() if isinstance(raw_features, torch.Tensor) else np.asarray(raw_features)
    if features.shape[0] != len(rows):
        raise FeatureCacheError("features row count does not match manifest")
    if not np.isfinite(features).all():
        raise FeatureCacheError("features contain non-finite values")
    if expected_feature_config is not None and cache.get("feature_config_hash") != hash_feature_config(expected_feature_config):
        raise FeatureCacheError("feature_config_hash is stale")

def write_feature_cache(cache: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); torch.save(dict(cache), p)


def read_feature_cache(path: str | Path) -> dict[str, Any]:
    loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise FeatureCacheError("feature cache file must contain a dictionary")
    return loaded


def load_and_validate_feature_cache(path: str | Path, manifest_rows: Sequence[Mapping[str, Any]], expected_feature_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cache = read_feature_cache(path)
    validate_feature_cache(cache, manifest_rows=manifest_rows, expected_feature_config=expected_feature_config)
    return cache

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_numpy_cache(path: str | Path, features: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, features.astype(np.float32, copy=False))


def load_numpy_cache(path: str | Path) -> np.ndarray:
    return np.load(Path(path)).astype(np.float32, copy=False)


def save_torch_cache(path: str | Path, features: np.ndarray, labels: np.ndarray, metadata: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": torch.from_numpy(features.astype(np.float32)), "labels": torch.from_numpy(labels.astype(np.int64)), "metadata": metadata}, p)


def write_feature_index(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path", "label", "generator", "split", "feature_path", "row_index"]
    mode = "a" if p.exists() else "w"
    with p.open(mode, newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})



def _row_identity(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("image_id") or row.get("path") or row.get("filepath") or row.get("rel_path") or "")


def save_split_features(output_dir: str | Path, *, feature_type: str, split: str, features: np.ndarray, labels: np.ndarray, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None = None) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / f"{feature_type}_{split}.npy"
    label_path = output / f"{feature_type}_{split}_labels.npy"
    row_path = output / f"{feature_type}_{split}_rows.csv"
    np.save(feature_path, np.asarray(features, dtype=np.float32))
    np.save(label_path, np.asarray(labels, dtype=np.int64))
    with row_path.open("w", newline="", encoding="utf-8") as row_file:
        row_writer = csv.DictWriter(row_file, fieldnames=["row_index", "sample_id", "path", "label", "generator", "split"])
        row_writer.writeheader()
        for index, row in enumerate(rows):
            row_writer.writerow({
                "row_index": index,
                "sample_id": _row_identity(row),
                "path": row.get("path") or row.get("filepath") or row.get("rel_path") or "",
                "label": row.get("label", ""),
                "generator": row.get("generator", ""),
                "split": row.get("split", split),
            })
    import json
    index_path = output / "feature_index.csv"
    exists = index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_type", "split", "feature_path", "label_path", "row_path", "sample_count", "feature_dim", "metadata"])
        if not exists:
            writer.writeheader()
        writer.writerow({"feature_type": feature_type, "split": split, "feature_path": feature_path.as_posix(), "label_path": label_path.as_posix(), "row_path": row_path.as_posix(), "sample_count": int(len(labels)), "feature_dim": int(features.shape[1]) if features.ndim == 2 else 0, "metadata": json.dumps(dict(metadata or {}), sort_keys=True)})
    return feature_path

def load_split_features(feature_dir: str | Path, feature_type: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    directory = Path(feature_dir)
    return np.load(directory / f"{feature_type}_{split}.npy"), np.load(directory / f"{feature_type}_{split}_labels.npy")



def load_split_feature_bundle(feature_dir: str | Path, feature_type: str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    directory = Path(feature_dir)
    features, labels = load_split_features(directory, feature_type, split)
    row_path = directory / f"{feature_type}_{split}_rows.csv"
    if not row_path.exists():
        raise FeatureCacheError(f"missing row identity file: {row_path}")
    with row_path.open("r", newline="", encoding="utf-8") as row_file:
        rows = list(csv.DictReader(row_file))
    if len(rows) != len(labels):
        raise FeatureCacheError(f"row identity count differs for {feature_type}/{split}: {len(rows)} != {len(labels)}")
    row_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(row_labels, labels.astype(np.int64)):
        raise FeatureCacheError(f"row labels differ from label cache for {feature_type}/{split}")
    return features, labels, rows


def assert_aligned_feature_rows(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]) -> None:
    left_ids = [_row_identity(row) for row in left_rows]
    right_ids = [_row_identity(row) for row in right_rows]
    if left_ids != right_ids:
        raise FeatureCacheError("feature cache sample_id/path order mismatch")
