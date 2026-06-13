from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

import src.eval.robustness as robustness_module
from src.data.make_dummy_dataset import make_dummy_dataset
from src.eval.robustness import ROBUSTNESS_COLUMNS, RobustnessError, apply_corruption, evaluate_robustness
from src.features.cache_features import cache_frequency_split
from src.models.checkpoint import save_checkpoint
from src.models.mlp_classifier import MLPClassifier
from src.train.common import TrainerSettings, train_feature_mlp


def test_apply_corruption_outputs_rgb_images_without_mutating_source(tiny_png: Path) -> None:
    before = _sha256(tiny_png)
    image = Image.open(tiny_png).convert("RGB")
    original_pixels = np.asarray(image).copy()

    jpeg = apply_corruption(image, "jpeg", "75")
    resized = apply_corruption(image, "resize", "0.5")
    blurred = apply_corruption(image, "blur", "1.0")
    legacy_resized = apply_corruption(image, "resize", "down_160")

    assert jpeg.mode == "RGB"
    assert resized.size == image.size
    assert blurred.mode == "RGB"
    assert legacy_resized.size == image.size
    assert np.array_equal(np.asarray(image), original_pixels)
    assert _sha256(tiny_png) == before


def test_frequency_robustness_writes_target_metrics_from_dataset_csv(tmp_path: Path) -> None:
    config = _trained_frequency_config(tmp_path)
    source_hashes = _source_hashes(_dataset_csv(config))

    result = evaluate_robustness(config, model_name="frequency_only", split="test")

    assert result.metrics_path == tmp_path / "reports" / "frequency_only_robustness_metrics.csv"
    rows = _read_csv(result.metrics_path)
    assert rows
    assert list(rows[0].keys()) == ROBUSTNESS_COLUMNS
    assert [(row["corruption"], row["severity"]) for row in rows] == [
        ("jpeg", "95"),
        ("jpeg", "75"),
        ("jpeg", "50"),
        ("resize", "0.5"),
        ("blur", "1.0"),
        ("blur", "2.0"),
    ]
    assert {row["corruption"] for row in rows} == {"jpeg", "resize", "blur"}
    assert {row["model_name"] for row in rows} == {"frequency_only"}
    assert all(0.0 <= float(row["accuracy"]) <= 1.0 for row in rows)
    assert all(row["roc_auc"] == "" or 0.0 <= float(row["roc_auc"]) <= 1.0 for row in rows)
    assert _source_hashes(_dataset_csv(config)) == source_hashes


def test_missing_selected_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    cache_frequency_split(config, split="test")
    checkpoint_path = tmp_path / "checkpoints" / "frequency_only.pt"

    try:
        evaluate_robustness(config, model_name="frequency_only", split="test")
    except RobustnessError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing checkpoint should fail")

    assert f"Missing checkpoint file: {checkpoint_path}" == message


def test_robustness_cli_missing_checkpoint_is_concise(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoints" / "frequency_only.pt"

    result = subprocess.run(
        [sys.executable, "-m", "src.eval.robustness", "--config", config_path.as_posix(), "--model", "frequency_only", "--split", "test"],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert "Robustness evaluation failed clearly" in output
    assert f"Missing checkpoint file: {checkpoint_path}" in output


def test_clip_only_robustness_uses_optional_clip_feature_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config(tmp_path)
    checkpoint_path = tmp_path / "checkpoints" / "clip_only.pt"
    model = MLPClassifier(input_dim=3, hidden_dim=4, dropout=0.0)
    save_checkpoint(
        checkpoint_path,
        model_state_dict=model.state_dict(),
        model_name="MLPClassifier",
        input_dim=3,
        hidden_dim=4,
        threshold=0.5,
        feature_type="clip",
        config_snapshot=config,
    )

    calls: list[int] = []

    def fake_clip_features(*_args: object, **kwargs: object) -> np.ndarray:
        labels = cast(np.ndarray, kwargs["labels"])
        calls.append(int(labels.shape[0]))
        return np.tile(np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32), (int(labels.shape[0]), 1))

    monkeypatch.setattr(robustness_module, "_extract_clip_feature_table", fake_clip_features)

    result = evaluate_robustness(config, model_name="clip_only", split="test")

    assert result.metrics_path.name == "clip_only_robustness_metrics.csv"
    assert calls == [6, 6, 6, 6, 6, 6]


def _trained_frequency_config(tmp_path: Path) -> dict[str, object]:
    config = _base_config(tmp_path)
    cache_frequency_split(config, split="train")
    cache_frequency_split(config, split="val")
    train_feature_mlp(config, TrainerSettings(feature_type="frequency", artifact_stem="frequency_only"))
    return config


def _base_config(tmp_path: Path) -> dict[str, object]:
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
        "clip": {"model_name": "ViT-B-32", "pretrained": "openai", "normalize_feature": True, "freeze": True},
        "frequency": {"method": "dct", "image_size": 24, "radial_bins": 10, "log_scale": True, "normalize_feature": True},
        "classifier": {"type": "mlp", "hidden_dim": 8, "dropout": 0.0, "num_classes": 1},
        "train": {"epochs": 3, "learning_rate": 0.01, "weight_decay": 0.0, "early_stopping_patience": 2, "loss": "bce_with_logits"},
        "eval": {"threshold": 0.5, "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "robustness": {"jpeg_qualities": [95, 75, 50], "resize_scales": [0.5], "blur_sigmas": [1.0, 2.0]},
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _source_hashes(csv_path: Path) -> dict[str, str]:
    return {row["image_id"]: _sha256(Path(row["filepath"])) for row in _read_csv(csv_path)}


def _dataset_csv(config: dict[str, object]) -> Path:
    paths = cast(dict[str, object], config["paths"])
    return Path(str(paths["dataset_csv"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
