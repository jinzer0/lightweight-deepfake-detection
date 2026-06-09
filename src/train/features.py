from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false, reportArgumentType=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.cache import FeatureCacheError, load_and_validate_feature_cache

FeatureMode = Literal["frequency_only", "clip_only", "fusion"]
SUPPORTED_FEATURE_MODES: tuple[FeatureMode, ...] = ("frequency_only", "clip_only", "fusion")


@dataclass(frozen=True)
class AssembledFeatures:
    mode: FeatureMode
    features: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    sample_ids: list[str]
    paths: list[str]
    transformers: dict[str, Any]
    metadata: dict[str, Any]


def assemble_features(
    mode: FeatureMode,
    manifest_rows: list[dict[str, Any]],
    *,
    frequency_cache_path: str | Path | None = None,
    clip_cache_path: str | Path | None = None,
) -> AssembledFeatures:
    if mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(f"mode must be one of {SUPPORTED_FEATURE_MODES}, got {mode!r}")

    frequency_cache = None
    clip_cache = None
    if mode in {"frequency_only", "fusion"}:
        if frequency_cache_path is None:
            raise ValueError(f"frequency_cache_path is required for mode {mode}")
        frequency_cache = load_and_validate_feature_cache(frequency_cache_path, manifest_rows=manifest_rows)
        _require_feature_type(frequency_cache, "frequency", "frequency_cache_path")
    if mode in {"clip_only", "fusion"}:
        if clip_cache_path is None:
            raise ValueError(f"clip_cache_path is required for mode {mode}")
        clip_cache = load_and_validate_feature_cache(clip_cache_path, manifest_rows=manifest_rows)
        _require_feature_type(clip_cache, "clip", "clip_cache_path")

    reference_cache = frequency_cache if frequency_cache is not None else clip_cache
    if reference_cache is None:
        raise ValueError("at least one feature cache is required")

    labels = np.asarray(reference_cache["labels"], dtype=np.int64)
    splits = np.asarray(reference_cache["splits"], dtype=object)
    train_mask = splits == "train"
    if not np.any(train_mask):
        raise ValueError("manifest must contain at least one train row")

    sample_ids = [str(value) for value in reference_cache["sample_ids"]]
    paths = [str(value) for value in reference_cache["paths"]]
    transformers: dict[str, Any] = {}
    branch_metadata: dict[str, Any] = {}

    if mode == "frequency_only":
        assert frequency_cache is not None
        features = _scale_frequency_branch(frequency_cache, train_mask, transformers)
        branch_metadata["frequency"] = _branch_metadata(frequency_cache)
    elif mode == "clip_only":
        assert clip_cache is not None
        features = np.asarray(clip_cache["features"], dtype=np.float32)
        branch_metadata["clip"] = _branch_metadata(clip_cache)
    else:
        assert frequency_cache is not None and clip_cache is not None
        _validate_fusion_alignment(frequency_cache, clip_cache)
        frequency_features = _scale_frequency_branch(frequency_cache, train_mask, transformers)
        clip_features = np.asarray(clip_cache["features"], dtype=np.float32)
        features = np.concatenate([frequency_features, clip_features], axis=1).astype(np.float32, copy=False)
        branch_metadata["frequency"] = _branch_metadata(frequency_cache)
        branch_metadata["clip"] = _branch_metadata(clip_cache)

    return AssembledFeatures(
        mode=mode,
        features=np.asarray(features, dtype=np.float32),
        labels=labels,
        splits=splits,
        train_mask=train_mask,
        val_mask=splits == "val",
        test_mask=splits == "test",
        sample_ids=sample_ids,
        paths=paths,
        transformers=transformers,
        metadata={
            "mode": mode,
            "feature_dim": int(features.shape[1]),
            "branches": branch_metadata,
        },
    )


def _scale_frequency_branch(cache: dict[str, Any], train_mask: np.ndarray, transformers: dict[str, Any]) -> np.ndarray:
    features = np.asarray(cache["features"], dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[train_mask])
    transformed = np.asarray(scaler.transform(features), dtype=np.float32)
    transformers["frequency_scaler"] = scaler
    if not np.isfinite(scaled).all() or not np.isfinite(transformed).all():
        raise FeatureCacheError("frequency scaling produced non-finite features")
    return transformed


def _require_feature_type(cache: dict[str, Any], expected: str, name: str) -> None:
    actual = str(cache["feature_type"])
    if actual != expected:
        raise FeatureCacheError(f"{name} must contain feature_type={expected}, got {actual}")


def _validate_fusion_alignment(frequency_cache: dict[str, Any], clip_cache: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for key in ("manifest_hash", "sample_ids", "paths", "labels", "splits"):
        left = frequency_cache[key]
        right = clip_cache[key]
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not np.array_equal(np.asarray(left), np.asarray(right)):
                mismatches.append(key)
        elif list(left) != list(right) if isinstance(left, list) and isinstance(right, list) else left != right:
            mismatches.append(key)
    if mismatches:
        raise FeatureCacheError(f"fusion cache alignment mismatch: {', '.join(mismatches)}")


def _branch_metadata(cache: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(cache["metadata"])
    return {
        "feature_type": str(cache["feature_type"]),
        "feature_config_hash": str(cache["feature_config_hash"]),
        "manifest_hash": str(cache["manifest_hash"]),
        "feature_dim": int(metadata["feature_dim"]),
        "normalization": str(metadata["normalization"]),
        "metadata": metadata,
    }
