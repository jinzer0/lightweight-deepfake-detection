from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from src.data.make_dummy_dataset import make_dummy_dataset
from src.features.cache_features import NpyFeatureCacheError, cache_frequency_split, cache_paths
from src.models.checkpoint import load_checkpoint
from src.train.common import TrainerSettings, train_feature_mlp


def test_frequency_training_smoke_writes_checkpoint_log_and_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    cache_frequency_split(config, split="train")
    cache_frequency_split(config, split="val")

    artifacts = train_feature_mlp(config, TrainerSettings(feature_type="frequency", artifact_stem="frequency_only"))

    assert artifacts.checkpoint_path.is_file()
    assert artifacts.train_log_path.is_file()
    assert artifacts.val_metrics_path.is_file()
    assert (tmp_path / "scalers" / "frequency_scaler.pkl").is_file()
    checkpoint = load_checkpoint(artifacts.checkpoint_path, expected_feature_type="frequency")
    assert checkpoint["model_name"] == "MLPClassifier"
    assert checkpoint["input_dim"] == 10
    assert checkpoint["hidden_dim"] == 8
    assert checkpoint["threshold"] == 0.5
    assert checkpoint["feature_type"] == "frequency"
    snapshot = cast(dict[str, Any], checkpoint["config_snapshot"])
    paths = cast(dict[str, Any], snapshot["paths"])
    assert paths["frequency_scaler_path"] == (tmp_path / "scalers" / "frequency_scaler.pkl").as_posix()

    log_text = artifacts.train_log_path.read_text(encoding="utf-8")
    assert "epoch,train_loss,val_loss,val_accuracy,val_precision,val_recall,val_f1,val_roc_auc" in log_text
    metrics = json.loads(artifacts.val_metrics_path.read_text(encoding="utf-8"))
    assert metrics["feature_type"] == "frequency"
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "loss"}.issubset(metrics["metrics"])


def test_training_missing_cache_error_names_train_features_and_labels(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = cache_paths(config, feature_type="frequency", split="train")

    with pytest.raises(NpyFeatureCacheError) as error:
        train_feature_mlp(config, TrainerSettings(feature_type="frequency", artifact_stem="frequency_only"))

    message = str(error.value)
    assert str(paths.features) in message
    assert str(paths.labels) in message
    assert "split='train'" in message


def test_frequency_training_cli_missing_cache_is_concise(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "missing-cache-config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    paths = cache_paths(config, feature_type="frequency", split="train")

    result = subprocess.run(
        [sys.executable, "-m", "src.train.train_frequency", "--config", config_path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert "Frequency training failed clearly" in output
    assert str(paths.features) in output
    assert str(paths.labels) in output


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
            "scaler_dir": (tmp_path / "scalers").as_posix(),
        },
        "data": {"image_size": 24, "batch_size": 8, "num_workers": 0},
        "frequency": {"method": "dct", "image_size": 24, "radial_bins": 10, "log_scale": True, "normalize_feature": True},
        "classifier": {"type": "mlp", "hidden_dim": 8, "dropout": 0.0, "num_classes": 1},
        "train": {"epochs": 3, "learning_rate": 0.01, "weight_decay": 0.0, "early_stopping_patience": 2, "loss": "bce_with_logits"},
        "eval": {"threshold": 0.5, "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]},
    }
