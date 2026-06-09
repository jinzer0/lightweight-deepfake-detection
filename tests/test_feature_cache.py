from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false

from pathlib import Path

import numpy as np
import pytest

from src.data.cifake import generate_manifest
from src.features.cache import (
    FeatureCacheError,
    build_metadata,
    create_feature_cache,
    load_and_validate_feature_cache,
    validate_feature_cache,
    write_feature_cache,
)
from src.features.frequency import DCT_POLICY, DEFAULT_FFT_EPSILON, DEFAULT_RADIAL_BINS, FrequencyFeatureConfig


def _manifest_rows(root: Path) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in generate_manifest(root, seed=42)]


def _metadata(feature_dim: int) -> dict[str, object]:
    return build_metadata(
        feature_dim=feature_dim,
        dtype="float32",
        normalization="raw_unscaled",
        seed=42,
        extra={
            "image_size": 224,
            "radial_bins": DEFAULT_RADIAL_BINS,
            "fft_epsilon": DEFAULT_FFT_EPSILON,
            "dct_policy": DCT_POLICY,
        },
        created_at="2026-06-07T00:00:00+00:00",
    )


def _cache(rows: list[dict[str, str]], feature_dim: int = 220) -> dict[str, object]:
    features = np.arange(len(rows) * feature_dim, dtype=np.float32).reshape(len(rows), feature_dim)
    config = FrequencyFeatureConfig().as_dict()
    return create_feature_cache(
        manifest_rows=rows,
        feature_type="frequency",
        feature_config=config,
        features=features,
        metadata=_metadata(feature_dim),
    )


def test_feature_cache_valid_round_trip(synthetic_cifake_root: Path, tmp_path: Path) -> None:
    rows = _manifest_rows(synthetic_cifake_root)
    cache = _cache(rows)
    cache_path = tmp_path / "features.pt"
    write_feature_cache(cache, cache_path)

    loaded = load_and_validate_feature_cache(cache_path, manifest_rows=rows, expected_feature_config=FrequencyFeatureConfig().as_dict())
    assert loaded["feature_type"] == "frequency"
    assert np.asarray(loaded["features"]).shape == (len(rows), 220)


def test_feature_cache_rejects_stale_manifest(synthetic_cifake_root: Path) -> None:
    rows = _manifest_rows(synthetic_cifake_root)
    cache = _cache(rows)
    stale_rows = [dict(row) for row in rows]
    stale_rows[0]["rel_path"] = "changed/path.png"
    with pytest.raises(FeatureCacheError, match="manifest_hash is stale"):
        validate_feature_cache(cache, manifest_rows=stale_rows)


def test_feature_cache_rejects_duplicate_missing_and_misaligned_rows(synthetic_cifake_root: Path) -> None:
    rows = _manifest_rows(synthetic_cifake_root)

    duplicate_cache = dict(_cache(rows))
    duplicate_ids = list(duplicate_cache["sample_ids"])
    duplicate_ids[1] = duplicate_ids[0]
    duplicate_cache["sample_ids"] = duplicate_ids
    with pytest.raises(FeatureCacheError, match="sample_ids contains duplicates"):
        validate_feature_cache(duplicate_cache, manifest_rows=rows)

    missing_cache = dict(_cache(rows))
    missing_cache["paths"] = list(missing_cache["paths"])[:-1]
    with pytest.raises(FeatureCacheError, match="paths length"):
        validate_feature_cache(missing_cache, manifest_rows=rows)

    misaligned_cache = dict(_cache(rows))
    paths = list(misaligned_cache["paths"])
    paths[0], paths[1] = paths[1], paths[0]
    misaligned_cache["paths"] = paths
    with pytest.raises(FeatureCacheError, match="paths order or values do not match"):
        validate_feature_cache(misaligned_cache, manifest_rows=rows)


def test_feature_cache_rejects_non_finite_features(synthetic_cifake_root: Path) -> None:
    rows = _manifest_rows(synthetic_cifake_root)
    cache = _cache(rows)
    features = np.asarray(cache["features"]).copy()
    features[0, 0] = np.nan
    cache["features"] = features
    with pytest.raises(FeatureCacheError, match="non-finite"):
        validate_feature_cache(cache, manifest_rows=rows)
