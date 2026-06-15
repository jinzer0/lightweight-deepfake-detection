from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false

import csv
from pathlib import Path

import numpy as np
import yaml

from src.data.manifest import write_manifest
from src.features.cache import build_metadata, create_feature_cache, write_feature_cache
from src.features.clip import CLIP_FEATURE_DIM, CLIP_NORMALIZATION
from src.features.frequency import DCT_BACKEND, DCT_POLICY, DEFAULT_FFT_EPSILON, DEFAULT_RADIAL_BINS, FrequencyFeatureConfig
from src.train.frequency_lr import PREDICTION_COLUMNS, train_classifier, verify_reload_equivalence


def _rows() -> list[dict[str, str]]:
    specs = [
        ("train-real", "real/train0.png", 0, "real", "train"),
        ("train-fake", "fake/train1.png", 1, "fake", "train"),
        ("val-real", "real/val0.png", 0, "real", "val"),
        ("val-fake", "fake/val1.png", 1, "fake", "val"),
        ("test-real", "real/test0.png", 0, "real", "test"),
        ("test-fake", "fake/test1.png", 1, "fake", "test"),
    ]
    return [
        {
            "sample_id": sample_id,
            "base_sample_id": sample_id,
            "rel_path": rel_path,
            "root": "/synthetic",
            "label": str(label),
            "class_name": class_name,
            "source": "synthetic",
            "source_split": split,
            "split": split,
            "width": "512",
            "height": "512",
            "sha256": f"{index:064x}",
            "file_size": "1",
            "mtime": "0",
            "status": "ok",
        }
        for index, (sample_id, rel_path, label, class_name, split) in enumerate(specs, start=1)
    ]


def _frequency_features(rows: list[dict[str, str]]) -> np.ndarray:
    features = np.zeros((len(rows), 220), dtype=np.float32)
    for index, row in enumerate(rows):
        sign = 1.0 if row["label"] == "1" else -1.0
        features[index, 0] = sign * 4.0
        features[index, 1] = sign * 2.0 + index * 0.01
    return features


def _clip_features(rows: list[dict[str, str]]) -> np.ndarray:
    features = np.zeros((len(rows), CLIP_FEATURE_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        sign = 1.0 if row["label"] == "1" else -1.0
        features[index, 0] = sign * 3.0
        features[index, 1] = sign * 1.5 + index * 0.01
    return features


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = _rows()
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, rows)

    frequency_metadata = build_metadata(
        feature_dim=220,
        dtype="float32",
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
    frequency_cache = create_feature_cache(
        manifest_rows=rows,
        feature_type="frequency",
        feature_config=FrequencyFeatureConfig().as_dict(),
        features=_frequency_features(rows),
        metadata=frequency_metadata,
    )
    frequency_path = tmp_path / "frequency.pt"
    write_feature_cache(frequency_cache, frequency_path)

    clip_metadata = build_metadata(
        feature_dim=CLIP_FEATURE_DIM,
        dtype="float32",
        normalization=CLIP_NORMALIZATION,
        seed=42,
        extra={"model_name": "test-clip", "preprocess_hash": "test-preprocess", "device": "cpu"},
        created_at="2026-06-07T00:00:00+00:00",
    )
    clip_cache = create_feature_cache(
        manifest_rows=rows,
        feature_type="clip",
        feature_config={"model_id": "test-clip"},
        features=_clip_features(rows),
        metadata=clip_metadata,
    )
    clip_path = tmp_path / "clip.pt"
    write_feature_cache(clip_cache, clip_path)
    return manifest_path, frequency_path, clip_path


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def test_logistic_regression_training_matrix_modes_have_prob_fake_and_reload(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)

    for mode in ["frequency_only", "clip_only", "fusion"]:
        result = train_classifier(
            manifest_path=manifest_path,
            output_dir=tmp_path / mode,
            mode=mode,
            classifier="logistic_regression",
            frequency_cache_path=frequency_path,
            clip_cache_path=clip_path,
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
        assert verify_reload_equivalence(
            manifest_path=manifest_path,
            output_dir=result.output_dir,
            mode=mode,
            frequency_cache_path=frequency_path,
            clip_cache_path=clip_path,
        ) <= 1e-12

        predictions = _prediction_rows(result.predictions_path)
        assert predictions[0].keys() == set(PREDICTION_COLUMNS)
        assert all(0.0 <= float(row["prob_fake"]) <= 1.0 for row in predictions)
        fake_probs = [float(row["prob_fake"]) for row in predictions if row["label"] == "1"]
        real_probs = [float(row["prob_fake"]) for row in predictions if row["label"] == "0"]
        assert min(fake_probs) > max(real_probs)

        with result.config_path.open("r", encoding="utf-8") as file_obj:
            config = yaml.safe_load(file_obj)
        assert config["mode"] == mode
        assert config["classifier"]["key"] == "logistic_regression"
        assert config["probability_supported"] is True
        assert config["decision_score_only"] is False
        assert config["streamlit_probability_eligible"] is True


def test_linear_svm_artifact_marks_decision_score_only_policy(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)

    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "svm",
        mode="fusion",
        classifier="linear_svm",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=500,
        verify_reload=True,
    )

    predictions = _prediction_rows(result.predictions_path)
    assert predictions[0].keys() == set(PREDICTION_COLUMNS)
    assert all(row["prob_fake"] == "" for row in predictions)
    assert all(row["score"] != "" for row in predictions)
    assert result.reload_max_abs_diff is not None
    assert result.reload_max_abs_diff <= 1e-12

    with result.config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    assert config["classifier"]["key"] == "linear_svm"
    assert config["probability_supported"] is False
    assert config["decision_score_only"] is True
    assert config["streamlit_probability_eligible"] is False
    assert config["calibration"]["probability_supported"] is False
