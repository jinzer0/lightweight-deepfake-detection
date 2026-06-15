from __future__ import annotations


import csv
import json
import warnings
from pathlib import Path

import numpy as np

from src.data.make_dummy_dataset import make_dummy_dataset
from src.eval.evaluate import PREDICTION_COLUMNS, evaluate_model, write_model_comparison
from src.eval.metrics import compute_binary_metrics
from src.features.cache_features import cache_frequency_split
from src.train.common import TrainerSettings, train_feature_mlp


def test_compute_binary_metrics_includes_confusion_matrix_and_auc() -> None:
    metrics = compute_binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.7, 0.8, 0.4]), threshold=0.5)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["roc_auc"] == 0.75


def test_compute_binary_metrics_one_class_warns_and_returns_null_auc() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = compute_binary_metrics(np.asarray([1, 1, 1]), np.asarray([0.2, 0.8, 0.9]))

    assert metrics["roc_auc"] is None
    assert any("one class" in str(item.message) for item in caught)


def test_evaluate_frequency_model_writes_required_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    cache_frequency_split(config, split="train")
    cache_frequency_split(config, split="val")
    cache_frequency_split(config, split="test")
    train_feature_mlp(config, TrainerSettings(feature_type="frequency", artifact_stem="frequency_only"))

    result = evaluate_model(config, model_name="frequency_only", split="test")

    assert result.metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert result.per_generator_path.is_file()
    assert result.comparison_path.is_file()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["model_name"] == "frequency_only"
    assert metrics["feature_type"] == "frequency"
    assert metrics["split"] == "test"
    assert metrics["sample_count"] == 4
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}.issubset(metrics["metrics"])

    with result.predictions_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    assert reader.fieldnames == PREDICTION_COLUMNS
    assert len(rows) == 4
    assert all(row["split"] == "test" for row in rows)
    assert all(0.0 <= float(row["pred_prob"]) <= 1.0 for row in rows)
    assert {row["model_name"] for row in rows} == {"frequency_only"}

    per_generator = list(csv.DictReader(result.per_generator_path.open("r", newline="", encoding="utf-8")))
    assert {row["generator"] for row in per_generator} == {"real_dummy", "dummy_generator"}
    assert all(row["roc_auc"] == "" for row in per_generator)

    comparison = list(csv.DictReader(result.comparison_path.open("r", newline="", encoding="utf-8")))
    assert [(row["model_name"], row["split"]) for row in comparison] == [("frequency_only", "test")]


def test_model_comparison_skips_unrelated_json(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "frequency_only_test_metrics.json").write_text(
        json.dumps({"model_name": "frequency_only", "split": "test", "sample_count": 2, "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": None}}),
        encoding="utf-8",
    )
    (report_dir / "legacy_metrics.json").write_text(json.dumps({"overall": {"accuracy": 0.0}}), encoding="utf-8")

    comparison_path = write_model_comparison(report_dir)

    rows = list(csv.DictReader(comparison_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["model_name"] == "frequency_only"


def _config(tmp_path: Path) -> dict[str, object]:
    csv_path = tmp_path / "metadata" / "dataset.csv"
    make_dummy_dataset(num_real=12, num_fake=12, output_dir=tmp_path / "images", csv_path=csv_path, width=24, height=24)
    return {
        "project": {"seed": 42, "device": "cpu"},
        "paths": {
            "dataset_csv": csv_path.as_posix(),
            "feature_dir": (tmp_path / "features").as_posix(),
            "checkpoint_dir": (tmp_path / "checkpoints").as_posix(),
            "report_dir": (tmp_path / "reports").as_posix(),
        },
        "data": {"image_size": 24, "batch_size": 8, "num_workers": 0},
        "frequency": {"method": "dct", "image_size": 24, "radial_bins": 10, "log_scale": True, "normalize_feature": True},
        "classifier": {"type": "mlp", "hidden_dim": 8, "dropout": 0.0, "num_classes": 1},
        "train": {"epochs": 3, "learning_rate": 0.01, "weight_decay": 0.0, "early_stopping_patience": 2, "loss": "bce_with_logits"},
        "eval": {"threshold": 0.5, "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    }
