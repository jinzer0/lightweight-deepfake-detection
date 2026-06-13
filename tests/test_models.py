from __future__ import annotations

# pyright: reportAny=false, reportPrivateLocalImportUsage=false, reportUnusedCallResult=false

from pathlib import Path
from typing import cast

import pytest
import torch

import src.models.checkpoint as checkpoint_module
from src.models.checkpoint import CheckpointError, load_checkpoint, save_checkpoint, validate_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.models.mlp_classifier import MLPClassifier


def test_mlp_classifier_outputs_batch_logits_for_bce_loss() -> None:
    model = MLPClassifier(input_dim=64, hidden_dim=16, dropout=0.1)
    features = torch.randn(4, 64)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    logits = model(features)
    loss = torch.nn.BCEWithLogitsLoss()(logits, targets)

    assert list(logits.shape) == [4]
    assert torch.isfinite(loss)


def test_mlp_classifier_validates_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        MLPClassifier(input_dim=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        MLPClassifier(input_dim=4, hidden_dim=0)
    with pytest.raises(ValueError, match="dropout"):
        MLPClassifier(input_dim=4, dropout=1.0)


def test_fusion_classifier_uses_clip_plus_frequency_input_dim() -> None:
    model = FusionClassifier(clip_dim=8, freq_dim=5, hidden_dim=7, dropout=0.0)
    features = torch.randn(3, 13)

    logits = model(features)

    assert model.input_dim == 13
    assert model.classifier.input_dim == 13
    assert list(logits.shape) == [3]


def test_checkpoint_round_trip_with_required_metadata(tmp_path: Path) -> None:
    model = MLPClassifier(input_dim=6, hidden_dim=4, dropout=0.0)
    checkpoint_path = tmp_path / "mlp.pt"

    save_checkpoint(
        checkpoint_path,
        model_state_dict=model.state_dict(),
        model_name="MLPClassifier",
        input_dim=6,
        hidden_dim=4,
        threshold=0.5,
        feature_type="clip",
        config_snapshot={"seed": 42},
        scaler_state={"mean": [0.0]},
    )

    loaded = load_checkpoint(checkpoint_path, expected_feature_type="clip")

    assert loaded["model_name"] == "MLPClassifier"
    assert loaded["input_dim"] == 6
    assert loaded["hidden_dim"] == 4
    assert loaded["threshold"] == 0.5
    assert loaded["feature_type"] == "clip"
    assert loaded["config_snapshot"] == {"seed": 42}
    assert loaded["scaler_state"] == {"mean": [0.0]}
    model_state_dict = loaded["model_state_dict"]
    assert isinstance(model_state_dict, dict)
    typed_model_state_dict = cast(dict[str, object], model_state_dict)
    assert set(typed_model_state_dict.keys()) == set(model.state_dict().keys())


def test_checkpoint_missing_key_error_names_missing_key() -> None:
    checkpoint: dict[str, object] = {
        "model_state_dict": {},
        "model_name": "MLPClassifier",
        "hidden_dim": 4,
        "threshold": 0.5,
        "feature_type": "clip",
        "config_snapshot": {},
    }

    with pytest.raises(CheckpointError, match="input_dim"):
        validate_checkpoint(checkpoint)


def test_checkpoint_load_prefers_cpu_weight_only_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {
            "model_state_dict": {},
            "model_name": "MLPClassifier",
            "input_dim": 6,
            "hidden_dim": 4,
            "threshold": 0.5,
            "feature_type": "clip",
            "config_snapshot": {},
        }

    checkpoint_path = tmp_path / "mlp.pt"
    monkeypatch.setattr(checkpoint_module.torch, "load", fake_load)

    loaded = load_checkpoint(checkpoint_path, expected_feature_type="clip")

    assert loaded["feature_type"] == "clip"
    assert calls == [{"path": checkpoint_path, "map_location": "cpu", "weights_only": True}]


def test_checkpoint_feature_type_mismatch_names_incompatible_values(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "mlp.pt"
    save_checkpoint(
        checkpoint_path,
        model_state_dict={},
        model_name="MLPClassifier",
        input_dim=6,
        hidden_dim=4,
        threshold=0.5,
        feature_type="frequency",
        config_snapshot={},
    )

    with pytest.raises(CheckpointError, match="feature_type incompatible: expected clip, got frequency"):
        load_checkpoint(checkpoint_path, expected_feature_type="clip")
