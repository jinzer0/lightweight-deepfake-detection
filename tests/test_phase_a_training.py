from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.local_manifest import generate_manifest
from src.data.manifest import write_manifest
from src.features.cache import build_metadata, create_feature_cache, write_feature_cache
from src.features.frequency import DCT_BACKEND, DCT_POLICY, DEFAULT_FFT_EPSILON, DEFAULT_RADIAL_BINS, FEATURE_DTYPE, FrequencyFeatureConfig, extract_frequency_features
from src.train.frequency_lr import PREDICTION_COLUMNS, train_frequency_logistic_regression, verify_reload_equivalence


def test_phase_a_frequency_training_reload_and_prediction_schema(synthetic_real_fake_root: Path, tmp_path: Path) -> None:
    rows = [{key: str(value) for key, value in row.items()} for row in generate_manifest(synthetic_real_fake_root, seed=42)]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)

    features = np.vstack([extract_frequency_features(Path(row["root"]) / row["rel_path"]) for row in rows]).astype(FEATURE_DTYPE)
    config = FrequencyFeatureConfig().as_dict()
    metadata = build_metadata(
        feature_dim=int(features.shape[1]),
        dtype=str(np.dtype(FEATURE_DTYPE).name),
        normalization="raw_unscaled",
        seed=42,
        extra={
            "image_size": 512,
            "radial_bins": DEFAULT_RADIAL_BINS,
            "fft_epsilon": DEFAULT_FFT_EPSILON,
            "dct_policy": DCT_POLICY,
            "dct_backend": DCT_BACKEND,
        },
        created_at="2026-06-07T00:00:00+00:00",
    )
    cache = create_feature_cache(manifest_rows=rows, feature_type="frequency", feature_config=config, features=features, metadata=metadata)
    cache_path = tmp_path / "frequency_cache.pt"
    write_feature_cache(cache, cache_path)

    output_dir = tmp_path / "phase_a"
    result = train_frequency_logistic_regression(
        manifest_path=manifest_path,
        feature_cache_path=cache_path,
        output_dir=output_dir,
        seed=42,
        max_iter=200,
        verify_reload=True,
    )

    assert result.model_path.exists()
    assert result.scaler_path.exists()
    assert result.config_path.exists()
    assert result.metrics_path.exists()
    assert result.predictions_path.exists()
    assert result.reload_max_abs_diff is not None
    assert result.reload_max_abs_diff <= 1e-12
    assert verify_reload_equivalence(manifest_path=manifest_path, feature_cache_path=cache_path, output_dir=output_dir) <= 1e-12

    with result.predictions_path.open("r", newline="", encoding="utf-8") as file_obj:
        prediction_rows = list(csv.DictReader(file_obj))
    assert prediction_rows
    assert prediction_rows[0].keys() == set(PREDICTION_COLUMNS)
    assert len(prediction_rows) == len(rows)
    assert all(0.0 <= float(row["prob_fake"]) <= 1.0 for row in prediction_rows)
    assert {int(row["label"]) for row in prediction_rows} == {0, 1}

    with result.metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)
    assert np.isfinite(metrics["splits"]["train"]["accuracy"])

    with result.config_path.open("r", encoding="utf-8") as file_obj:
        config_snapshot = yaml.safe_load(file_obj)
    assert config_snapshot["mode"] == "frequency_only"
    assert config_snapshot["classifier"]["type"] == "LogisticRegression"
