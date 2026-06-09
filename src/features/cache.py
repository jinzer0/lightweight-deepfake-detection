from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedCallResult=false, reportMissingImports=false

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..data.manifest import CLASS_TO_LABEL, MANIFEST_COLUMNS


SCHEMA_VERSION = "feature_cache_v1"
FEATURE_TYPES = {"clip", "frequency"}
REQUIRED_KEYS = {
    "schema_version",
    "manifest_hash",
    "feature_type",
    "feature_config_hash",
    "sample_ids",
    "paths",
    "labels",
    "splits",
    "features",
    "metadata",
}
BASE_METADATA_KEYS = {"feature_dim", "dtype", "normalization", "label_mapping", "created_at", "seed"}
CLIP_METADATA_KEYS = {"model_name", "preprocess_hash", "device"}
FREQUENCY_METADATA_KEYS = {"image_size", "radial_bins", "fft_epsilon", "dct_policy"}
CORRUPTION_METADATA_KEYS = {"base_sample_ids", "corruption_type", "corruption_level", "clean_manifest_hash"}


class FeatureCacheError(ValueError):
    pass


class TorchDependencyError(ImportError):
    pass


def hash_manifest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [_normalize_manifest_row(row) for row in rows]
    return _stable_hash(normalized)


def hash_feature_config(config: Mapping[str, Any]) -> str:
    return _stable_hash(_json_ready(config))


def create_feature_cache(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    feature_type: str,
    feature_config: Mapping[str, Any],
    features: Any,
    metadata: Mapping[str, Any],
    manifest_hash: str | None = None,
    feature_config_hash: str | None = None,
) -> dict[str, Any]:
    resolved_feature_config_hash = feature_config_hash or hash_feature_config(feature_config)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest_hash or hash_manifest_rows(manifest_rows),
        "feature_type": feature_type,
        "feature_config_hash": resolved_feature_config_hash,
        "sample_ids": [str(row.get("sample_id", "")) for row in manifest_rows],
        "paths": [str(row.get("rel_path", "")) for row in manifest_rows],
        "labels": np.asarray([_row_label(row) for row in manifest_rows], dtype=np.int64),
        "splits": [str(row.get("split", "")) for row in manifest_rows],
        "features": features,
        "metadata": dict(metadata),
    }
    validate_feature_cache(
        cache,
        manifest_rows=manifest_rows,
        expected_feature_config_hash=resolved_feature_config_hash,
    )
    return cache


def write_feature_cache(cache: Mapping[str, Any], path: str | Path) -> None:
    torch = _require_torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), output_path)


def read_feature_cache(path: str | Path) -> dict[str, Any]:
    torch = _require_torch()
    try:
        loaded = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(Path(path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise FeatureCacheError("feature cache file did not contain a dictionary")
    return loaded


def load_and_validate_feature_cache(
    path: str | Path,
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    expected_feature_config: Mapping[str, Any] | None = None,
    expected_feature_config_hash: str | None = None,
) -> dict[str, Any]:
    cache = read_feature_cache(path)
    validate_feature_cache(
        cache,
        manifest_rows=manifest_rows,
        expected_feature_config=expected_feature_config,
        expected_feature_config_hash=expected_feature_config_hash,
    )
    return cache


def validate_feature_cache(
    cache: Mapping[str, Any],
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    expected_feature_config: Mapping[str, Any] | None = None,
    expected_feature_config_hash: str | None = None,
) -> None:
    errors: list[str] = []
    _validate_required_cache_keys(cache, errors)
    if errors:
        raise FeatureCacheError("; ".join(errors))

    if cache["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}, got {cache['schema_version']}")
    if cache["feature_type"] not in FEATURE_TYPES:
        errors.append(f"feature_type must be one of {sorted(FEATURE_TYPES)}, got {cache['feature_type']}")

    manifest_hash = hash_manifest_rows(manifest_rows)
    if cache["manifest_hash"] != manifest_hash:
        errors.append("manifest_hash is stale or does not match manifest rows")

    config_hash = expected_feature_config_hash
    if expected_feature_config is not None:
        config_hash = hash_feature_config(expected_feature_config)
    if config_hash is not None and cache["feature_config_hash"] != config_hash:
        errors.append("feature_config_hash is stale or does not match expected config")

    sample_ids = _string_list(cache["sample_ids"], "sample_ids", errors)
    paths = _string_list(cache["paths"], "paths", errors)
    splits = _string_list(cache["splits"], "splits", errors)
    labels = _array(cache["labels"], "labels", errors)
    features = _array(cache["features"], "features", errors)
    metadata = cache["metadata"]
    if not isinstance(metadata, Mapping):
        errors.append("metadata must be a mapping")
        metadata = {}

    row_count = len(manifest_rows)
    lengths = {
        "sample_ids": len(sample_ids),
        "paths": len(paths),
        "splits": len(splits),
    }
    if labels is not None:
        lengths["labels"] = int(labels.shape[0]) if labels.ndim > 0 else 0
    if features is not None:
        lengths["features"] = int(features.shape[0]) if features.ndim > 0 else 0
    for name, length in lengths.items():
        if length != row_count:
            errors.append(f"{name} length {length} does not match manifest row count {row_count}")

    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample_ids contains duplicates")
    expected_ids = [str(row.get("sample_id", "")) for row in manifest_rows]
    expected_paths = [str(row.get("rel_path", "")) for row in manifest_rows]
    expected_labels = [_row_label(row) for row in manifest_rows]
    expected_splits = [str(row.get("split", "")) for row in manifest_rows]
    _compare_sequence("sample_ids", sample_ids, expected_ids, errors)
    _compare_sequence("paths", paths, expected_paths, errors)
    _compare_sequence("splits", splits, expected_splits, errors)
    if labels is not None and labels.ndim == 1:
        _compare_sequence("labels", labels.astype(int).tolist(), expected_labels, errors)
    elif labels is not None:
        errors.append(f"labels must be one-dimensional, got shape {labels.shape}")

    if features is not None:
        if features.ndim != 2:
            errors.append(f"features must be two-dimensional, got shape {features.shape}")
        elif not np.isfinite(features).all():
            errors.append("features contains non-finite values")

    _validate_metadata(metadata, str(cache["feature_type"]), features, errors)

    if errors:
        raise FeatureCacheError("; ".join(errors))


def build_metadata(
    *,
    feature_dim: int,
    dtype: str,
    normalization: str,
    seed: int,
    extra: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "feature_dim": int(feature_dim),
        "dtype": str(dtype),
        "normalization": str(normalization),
        "label_mapping": dict(CLASS_TO_LABEL),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
    }
    metadata.update(dict(extra))
    return metadata


def _validate_required_cache_keys(cache: Mapping[str, Any], errors: list[str]) -> None:
    missing = sorted(REQUIRED_KEYS - set(cache.keys()))
    if missing:
        errors.append(f"cache missing required keys: {', '.join(missing)}")
    extra = sorted(set(cache.keys()) - REQUIRED_KEYS)
    if extra:
        errors.append(f"cache has unexpected top-level keys: {', '.join(extra)}")


def _validate_metadata(metadata: Mapping[str, Any], feature_type: str, features: np.ndarray | None, errors: list[str]) -> None:
    missing = sorted(BASE_METADATA_KEYS - set(metadata.keys()))
    if missing:
        errors.append(f"metadata missing required keys: {', '.join(missing)}")
    if metadata.get("label_mapping") != CLASS_TO_LABEL:
        errors.append(f"metadata label_mapping must be {CLASS_TO_LABEL}")

    feature_keys = CLIP_METADATA_KEYS if feature_type == "clip" else FREQUENCY_METADATA_KEYS if feature_type == "frequency" else set()
    missing_feature_keys = sorted(feature_keys - set(metadata.keys()))
    if missing_feature_keys:
        errors.append(f"{feature_type} metadata missing required keys: {', '.join(missing_feature_keys)}")

    corruption_declared = any(key in metadata for key in CORRUPTION_METADATA_KEYS) or bool(metadata.get("is_corrupted"))
    if corruption_declared:
        missing_corruption_keys = sorted(CORRUPTION_METADATA_KEYS - set(metadata.keys()))
        if missing_corruption_keys:
            errors.append(f"corrupted cache metadata missing required keys: {', '.join(missing_corruption_keys)}")

    if features is not None and features.ndim == 2:
        try:
            feature_dim = int(metadata.get("feature_dim"))
        except (TypeError, ValueError):
            errors.append(f"metadata feature_dim must be an integer, got {metadata.get('feature_dim')}")
        else:
            if feature_dim != int(features.shape[1]):
                errors.append(f"metadata feature_dim {feature_dim} does not match features.shape[1] {features.shape[1]}")

    if not metadata.get("dtype"):
        errors.append("metadata dtype must be non-empty")
    if not metadata.get("normalization"):
        errors.append("metadata normalization must be non-empty")
    if not metadata.get("created_at"):
        errors.append("metadata created_at must be non-empty")
    if "seed" in metadata:
        try:
            int(metadata["seed"])
        except (TypeError, ValueError):
            errors.append(f"metadata seed must be an integer, got {metadata['seed']}")


def _compare_sequence(name: str, actual: Sequence[Any], expected: Sequence[Any], errors: list[str]) -> None:
    if list(actual) == list(expected):
        return
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"{name} missing manifest values: {missing[:5]}")
    if extra:
        errors.append(f"{name} has unexpected values: {extra[:5]}")
    if not missing and not extra:
        errors.append(f"{name} order or values do not match manifest")


def _string_list(value: Any, name: str, errors: list[str]) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return []
    if not all(isinstance(item, str) for item in value):
        errors.append(f"{name} must contain only strings")
    return list(value)


def _array(value: Any, name: str, errors: list[str]) -> np.ndarray | None:
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        errors.append(f"{name} cannot be converted to an array: {exc}")
        return None


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")


def _row_label(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("label", ""))
    except (TypeError, ValueError) as exc:
        raise FeatureCacheError(f"manifest row has non-integer label for sample_id {row.get('sample_id', '')}: {row.get('label', '')}") from exc


def _normalize_manifest_row(row: Mapping[str, Any]) -> dict[str, str]:
    columns = MANIFEST_COLUMNS if all(column in row for column in MANIFEST_COLUMNS) else sorted(str(key) for key in row.keys())
    return {column: str(row.get(column, "")) for column in columns}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TorchDependencyError("torch is required to read or write .pt feature caches; install requirements.txt or use in-memory validation") from exc
    return torch
