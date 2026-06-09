from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false

from pathlib import Path

import numpy as np
import pytest

from src.features.cache import FeatureCacheError, build_metadata, create_feature_cache, hash_manifest_rows, write_feature_cache
from src.features.clip import CLIP_FEATURE_DIM, CLIP_NORMALIZATION
from src.features.frequency import (
    DCT_BACKEND,
    DCT_POLICY,
    DEFAULT_FFT_EPSILON,
    DEFAULT_RADIAL_BINS,
    FrequencyFeatureConfig,
)
from src.train.features import assemble_features


def _manifest_rows() -> list[dict[str, str]]:
    return [
        {"sample_id": "train-real", "rel_path": "real/train.png", "label": "0", "split": "train"},
        {"sample_id": "train-fake", "rel_path": "fake/train.png", "label": "1", "split": "train"},
        {"sample_id": "val-real", "rel_path": "real/val.png", "label": "0", "split": "val"},
        {"sample_id": "test-fake", "rel_path": "fake/test.png", "label": "1", "split": "test"},
    ]


def _frequency_cache(rows: list[dict[str, str]]) -> dict[str, object]:
    features = np.arange(len(rows) * 220, dtype=np.float32).reshape(len(rows), 220)
    metadata = build_metadata(
        feature_dim=220,
        dtype="float32",
        normalization="raw_unscaled",
        seed=42,
        extra={
            "image_size": 224,
            "radial_bins": DEFAULT_RADIAL_BINS,
            "fft_epsilon": DEFAULT_FFT_EPSILON,
            "dct_policy": DCT_POLICY,
            "dct_backend": DCT_BACKEND,
        },
        created_at="2026-06-07T00:00:00+00:00",
    )
    return create_feature_cache(
        manifest_rows=rows,
        feature_type="frequency",
        feature_config=FrequencyFeatureConfig().as_dict(),
        features=features,
        metadata=metadata,
    )


def _clip_cache(rows: list[dict[str, str]]) -> dict[str, object]:
    features = np.tile(np.linspace(0.0, 1.0, CLIP_FEATURE_DIM, dtype=np.float32), (len(rows), 1))
    metadata = build_metadata(
        feature_dim=CLIP_FEATURE_DIM,
        dtype="float32",
        normalization=CLIP_NORMALIZATION,
        seed=42,
        extra={
            "model_name": "test-clip",
            "preprocess_hash": "test-preprocess",
            "device": "cpu",
        },
        created_at="2026-06-07T00:00:00+00:00",
    )
    return create_feature_cache(
        manifest_rows=rows,
        feature_type="clip",
        feature_config={"model_id": "test-clip", "feature_dim": CLIP_FEATURE_DIM, "normalization": CLIP_NORMALIZATION},
        features=features,
        metadata=metadata,
    )


def _write_caches(tmp_path: Path) -> tuple[list[dict[str, str]], Path, Path]:
    rows = _manifest_rows()
    frequency_path = tmp_path / "frequency.pt"
    clip_path = tmp_path / "clip.pt"
    write_feature_cache(_frequency_cache(rows), frequency_path)
    write_feature_cache(_clip_cache(rows), clip_path)
    return rows, frequency_path, clip_path


def test_assemble_single_branch_modes_preserve_metadata_and_masks(tmp_path: Path) -> None:
    rows, frequency_path, clip_path = _write_caches(tmp_path)

    frequency = assemble_features("frequency_only", rows, frequency_cache_path=frequency_path)
    clip = assemble_features("clip_only", rows, clip_cache_path=clip_path)

    assert frequency.features.shape == (4, 220)
    assert clip.features.shape == (4, CLIP_FEATURE_DIM)
    assert frequency.train_mask.tolist() == [True, True, False, False]
    assert frequency.val_mask.tolist() == [False, False, True, False]
    assert frequency.test_mask.tolist() == [False, False, False, True]
    assert frequency.sample_ids == [row["sample_id"] for row in rows]
    assert frequency.paths == [row["rel_path"] for row in rows]
    np.testing.assert_allclose(frequency.features[frequency.train_mask].mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(frequency.features[~frequency.train_mask].mean(axis=0), 0.0, atol=1e-6)
    assert "frequency_scaler" in frequency.transformers
    assert clip.transformers == {}


def test_assemble_fusion_validates_alignment_and_concatenates_branch_features(tmp_path: Path) -> None:
    rows, frequency_path, clip_path = _write_caches(tmp_path)

    assembled = assemble_features("fusion", rows, frequency_cache_path=frequency_path, clip_cache_path=clip_path)

    assert assembled.features.shape == (4, 220 + CLIP_FEATURE_DIM)
    assert assembled.metadata["feature_dim"] == 220 + CLIP_FEATURE_DIM
    assert assembled.metadata["branches"]["frequency"]["normalization"] == "raw_unscaled"
    assert assembled.metadata["branches"]["clip"]["normalization"] == CLIP_NORMALIZATION
    np.testing.assert_allclose(assembled.features[assembled.train_mask, :220].mean(axis=0), 0.0, atol=1e-6)


def test_assemble_fusion_rejects_cache_alignment_mismatch(tmp_path: Path) -> None:
    rows = _manifest_rows()
    frequency_path = tmp_path / "frequency.pt"
    clip_path = tmp_path / "clip.pt"
    write_feature_cache(_frequency_cache(rows), frequency_path)
    clip_cache = _clip_cache(rows)
    clip_cache["sample_ids"] = ["wrong-id", *list(clip_cache["sample_ids"])[1:]]
    clip_cache["manifest_hash"] = hash_manifest_rows(rows)
    write_feature_cache(clip_cache, clip_path)

    with pytest.raises(FeatureCacheError, match="sample_ids"):
        assemble_features("fusion", rows, frequency_cache_path=frequency_path, clip_cache_path=clip_path)
