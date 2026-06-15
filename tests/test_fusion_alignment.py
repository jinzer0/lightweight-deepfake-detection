from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.features.cache_features import NpyFeatureCacheError, write_feature_cache
from src.models.checkpoint import load_checkpoint
from src.train.train_fusion import align_feature_tables, train_fusion


def test_reorders_by_image_id(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_split(config, split="train", frequency_order=[2, 0, 3, 1])

    aligned = align_feature_tables(config, split="train")

    assert aligned.meta["image_id"].tolist() == ["train-real-a", "train-fake-a", "train-real-b", "train-fake-b"]
    np.testing.assert_array_equal(aligned.labels, np.asarray([0, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(aligned.clip_features, _clip_features(4))
    np.testing.assert_array_equal(aligned.frequency_features, _frequency_features(4))
    np.testing.assert_array_equal(aligned.features, np.concatenate([_clip_features(4), _frequency_features(4)], axis=1))


def test_label_mismatch_raises(tmp_path: Path) -> None:
    config = _config(tmp_path)
    meta = _meta("train")
    frequency_meta = meta.copy()
    frequency_meta.loc[frequency_meta["image_id"] == "train-fake-a", "label"] = 0
    write_feature_cache(config, feature_type="clip", split="train", features=_clip_features(4), labels=np.asarray([0, 1, 0, 1]), meta=meta)
    write_feature_cache(config, feature_type="frequency", split="train", features=_frequency_features(4), labels=np.asarray([0, 0, 0, 1]), meta=frequency_meta)

    with pytest.raises(NpyFeatureCacheError, match="image_id='train-fake-a'.*clip=1 frequency=0"):
        align_feature_tables(config, split="train")


def test_missing_image_id_names_side() -> None:
    clip_meta = _meta("train")
    frequency_meta = clip_meta.copy()
    frequency_meta.loc[0, "image_id"] = "train-frequency-extra"

    with pytest.raises(NpyFeatureCacheError) as error:
        align_feature_tables(
            split="train",
            clip_features=_clip_features(4),
            clip_labels=np.asarray([0, 1, 0, 1]),
            clip_meta=clip_meta,
            frequency_features=_frequency_features(4),
            frequency_labels=np.asarray([0, 1, 0, 1]),
            frequency_meta=frequency_meta,
        )

    message = str(error.value)
    assert "missing from frequency" in message
    assert "train-real-a" in message
    assert "missing from clip" in message
    assert "train-frequency-extra" in message


def test_duplicate_image_id_raises() -> None:
    clip_meta = _meta("train")
    clip_meta.loc[1, "image_id"] = clip_meta.loc[0, "image_id"]

    with pytest.raises(NpyFeatureCacheError, match="clip metadata image_id contains duplicate"):
        align_feature_tables(
            split="train",
            clip_features=_clip_features(4),
            clip_labels=np.asarray([0, 1, 0, 1]),
            clip_meta=clip_meta,
            frequency_features=_frequency_features(4),
            frequency_labels=np.asarray([0, 1, 0, 1]),
            frequency_meta=_meta("train"),
        )


def test_train_fusion_writes_artifacts_with_fusion_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_split(config, split="train", frequency_order=[1, 3, 0, 2])
    _write_split(config, split="val", frequency_order=[1, 0])

    artifacts = train_fusion(config)

    assert artifacts.checkpoint_path == tmp_path / "checkpoints" / "fusion.pt"
    assert artifacts.checkpoint_path.is_file()
    assert (tmp_path / "scalers" / "frequency_scaler.pkl").is_file()
    assert artifacts.train_log_path == tmp_path / "reports" / "fusion_train_log.csv"
    assert artifacts.val_metrics_path == tmp_path / "reports" / "fusion_val_metrics.json"
    checkpoint = load_checkpoint(artifacts.checkpoint_path, expected_feature_type="fusion")
    assert checkpoint["model_name"] == "FusionClassifier"
    assert checkpoint["input_dim"] == 5
    snapshot = cast(dict[str, Any], checkpoint["config_snapshot"])
    paths = cast(dict[str, Any], snapshot["paths"])
    assert paths["frequency_scaler_path"] == (tmp_path / "scalers" / "frequency_scaler.pkl").as_posix()
    metrics = json.loads(artifacts.val_metrics_path.read_text(encoding="utf-8"))
    assert metrics["feature_type"] == "fusion"
    assert metrics["model_name"] == "FusionClassifier"


def test_fusion_cli_missing_cache_is_concise(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "src.train.train_fusion", "--config", config_path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert "Fusion training failed clearly" in output
    assert "requires clip cache" in output
    assert "train_features.npy" in output
    assert "train_labels.npy" in output


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project": {"seed": 42, "device": "cpu"},
        "paths": {
            "dataset_csv": (tmp_path / "metadata" / "dataset.csv").as_posix(),
            "feature_dir": (tmp_path / "features").as_posix(),
            "checkpoint_dir": (tmp_path / "checkpoints").as_posix(),
            "report_dir": (tmp_path / "reports").as_posix(),
            "scaler_dir": (tmp_path / "scalers").as_posix(),
        },
        "data": {"batch_size": 2},
        "classifier": {"hidden_dim": 4, "dropout": 0.0},
        "train": {"epochs": 3, "learning_rate": 0.01, "weight_decay": 0.0, "early_stopping_patience": 2},
        "eval": {"threshold": 0.5},
    }


def _write_split(config: dict[str, object], *, split: str, frequency_order: list[int]) -> None:
    row_count = len(frequency_order)
    meta = _meta(split).iloc[:row_count].reset_index(drop=True)
    labels = meta["label"].to_numpy(dtype=np.int64)
    write_feature_cache(config, feature_type="clip", split=split, features=_clip_features(row_count), labels=labels, meta=meta)
    frequency_meta = meta.iloc[frequency_order].reset_index(drop=True)
    frequency_labels = labels[frequency_order]
    frequency_features = _frequency_features(row_count)[frequency_order]
    write_feature_cache(config, feature_type="frequency", split=split, features=frequency_features, labels=frequency_labels, meta=frequency_meta)


def _meta(split: str) -> pd.DataFrame:
    labels = [0, 1, 0, 1]
    suffixes = ["real-a", "fake-a", "real-b", "fake-b"]
    return pd.DataFrame(
        {
            "image_id": [f"{split}-{suffix}" for suffix in suffixes],
            "filepath": [f"/tmp/{split}-{suffix}.png" for suffix in suffixes],
            "label": labels,
            "class_name": ["real", "fake", "real", "fake"],
            "dataset": ["unit"] * 4,
            "generator": ["camera", "synthetic", "camera", "synthetic"],
            "split": [split] * 4,
            "width": [16] * 4,
            "height": [16] * 4,
            "ext": [".png"] * 4,
        }
    )


def _clip_features(row_count: int) -> np.ndarray:
    return np.arange(row_count * 3, dtype=np.float32).reshape(row_count, 3)


def _frequency_features(row_count: int) -> np.ndarray:
    return (100 + np.arange(row_count * 2, dtype=np.float32)).reshape(row_count, 2)
