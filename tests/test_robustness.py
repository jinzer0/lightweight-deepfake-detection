from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportArgumentType=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnusedParameter=false, reportUnusedCallResult=false

import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from src.data.cifake import generate_manifest
from src.data.manifest import write_manifest
from src.eval.robustness import apply_corruption, run_frequency_robustness
from src.features.cache import build_metadata, create_feature_cache, hash_manifest_file, read_feature_cache, write_feature_cache
from src.features.frequency import (
    DCT_BACKEND,
    DCT_POLICY,
    DEFAULT_FFT_EPSILON,
    DEFAULT_RADIAL_BINS,
    FEATURE_DTYPE,
    FrequencyFeatureConfig,
    extract_frequency_features,
)
from src.train.frequency_lr import train_frequency_logistic_regression


class NoFitScaler:
    def __init__(self, scaler: Any) -> None:
        self.scaler = scaler

    def fit(self, features: Any, labels: Any | None = None) -> None:
        raise AssertionError("robustness must not fit scaler")

    def fit_transform(self, features: Any, labels: Any | None = None) -> None:
        raise AssertionError("robustness must not fit_transform scaler")

    def transform(self, features: Any) -> Any:
        return self.scaler.transform(features)


def test_apply_corruption_outputs_rgb_images(tiny_png: Path) -> None:
    image = Image.open(tiny_png).convert("RGB")

    jpeg = apply_corruption(image, "jpeg", "quality_75")
    resized = apply_corruption(image, "resize", "down_160")
    blurred = apply_corruption(image, "blur", "sigma_1.0")

    assert jpeg.mode == "RGB"
    assert resized.size == (224, 224)
    assert blurred.mode == "RGB"
    assert np.asarray(blurred).shape[-1] == 3


def test_quick_frequency_robustness_writes_metrics_and_summary(synthetic_cifake_root: Path, tmp_path: Path) -> None:
    manifest_path, experiment_dir = _train_tiny_frequency_artifact(synthetic_cifake_root, tmp_path)
    output_dir = tmp_path / "robustness"

    result = run_frequency_robustness(
        manifest_path=manifest_path,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        mode="quick",
    )

    assert result.metrics_path.stat().st_size > 0
    assert result.summary_path.stat().st_size > 0
    rows = _read_csv(result.metrics_path)
    assert [(row["corruption_type"], row["corruption_level"]) for row in rows] == [
        ("jpeg", "quality_75"),
        ("resize", "down_160"),
        ("blur", "sigma_1.0"),
    ]
    assert {row["base_sample_ids"] for row in rows}
    assert all(row["clean_accuracy"] for row in rows)
    assert all(Path(row["model_path"]).exists() for row in rows)
    assert all(Path(row["corrupted_cache_path"]).exists() for row in rows)

    clean_manifest_hash = hash_manifest_file(manifest_path)
    for row in rows:
        cache = read_feature_cache(row["corrupted_cache_path"])
        metadata = cache["metadata"]
        assert cache["schema_version"] == "feature_cache_v1"
        assert cache["feature_type"] == "frequency"
        assert metadata["base_sample_ids"]
        assert metadata["corruption_type"] == row["corruption_type"]
        assert metadata["corruption_level"] == row["corruption_level"]
        assert metadata["clean_manifest_hash"] == clean_manifest_hash
        assert np.asarray(cache["features"]).shape[0] == int(row["sample_count"])


def test_robustness_reuses_scaler_transform_without_refit(synthetic_cifake_root: Path, tmp_path: Path) -> None:
    manifest_path, experiment_dir = _train_tiny_frequency_artifact(synthetic_cifake_root, tmp_path)
    transformers = joblib.load(experiment_dir / "scaler.joblib")
    transformers["frequency_scaler"] = NoFitScaler(transformers["frequency_scaler"])
    _ = joblib.dump(transformers, experiment_dir / "scaler.joblib")

    result = run_frequency_robustness(
        manifest_path=manifest_path,
        experiment_dir=experiment_dir,
        output_dir=tmp_path / "robustness_no_refit",
        mode="quick",
        max_samples=2,
    )

    assert len(result.rows) == 3


def _train_tiny_frequency_artifact(data_root: Path, tmp_path: Path) -> tuple[Path, Path]:
    rows = [{key: str(value) for key, value in row.items()} for row in generate_manifest(data_root, seed=42)]
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)
    features = np.vstack([extract_frequency_features(Path(row["root"]) / row["rel_path"]) for row in rows]).astype(FEATURE_DTYPE)
    metadata = build_metadata(
        feature_dim=int(features.shape[1]),
        dtype=str(np.dtype(FEATURE_DTYPE).name),
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
    cache = create_feature_cache(
        manifest_rows=rows,
        feature_type="frequency",
        feature_config=FrequencyFeatureConfig().as_dict(),
        features=features,
        metadata=metadata,
    )
    cache_path = tmp_path / "frequency.pt"
    write_feature_cache(cache, cache_path)
    result = train_frequency_logistic_regression(
        manifest_path=manifest_path,
        feature_cache_path=cache_path,
        output_dir=tmp_path / "frequency_lr",
        max_iter=200,
        verify_reload=False,
    )
    return manifest_path, result.output_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))
