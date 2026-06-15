from __future__ import annotations


import json
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.inference.detector_service import DetectorService
from src.models.checkpoint import save_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.models.mlp_classifier import MLPClassifier


def test_frequency_only_detector_returns_scores_and_visualizations(tmp_path: Path, tiny_png: Path) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _write_mlp_checkpoint(config, "frequency_only", feature_type="frequency", input_dim=10, bias=0.8)

    result = DetectorService(config_path=config_path, model_name="frequency_only").predict(tiny_png)

    assert list(result) == [
        "ai_prob",
        "pred_label",
        "confidence",
        "clip_score",
        "frequency_score",
        "fusion_score",
        "spectrum_path",
        "radial_spectrum_path",
    ]
    assert result["ai_prob"] == pytest.approx(torch.sigmoid(torch.tensor(0.8)).item())
    assert result["pred_label"] == "AI"
    assert result["confidence"] == "medium"
    assert result["clip_score"] is None
    assert result["frequency_score"] == result["ai_prob"]
    assert result["fusion_score"] is None
    assert Path(str(result["spectrum_path"])).is_file()
    assert Path(str(result["radial_spectrum_path"])).is_file()
    assert Path(str(result["spectrum_path"])).parent == tmp_path / "figures"
    assert ".." not in Path(str(result["spectrum_path"])).name


def test_fusion_detector_warns_and_returns_none_for_missing_branch_checkpoint(
    tmp_path: Path,
    tiny_png: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _write_fusion_checkpoint(config, bias=-1.0)
    _write_mlp_checkpoint(config, "frequency_only", feature_type="frequency", input_dim=10, bias=0.5)

    monkeypatch.setattr("src.inference.detector_service.load_clip_model_and_preprocess", lambda _config, device: (_FakeClipModel(), _fake_clip_preprocess))

    detector = DetectorService(config_path=config_path, model_name="fusion")
    with pytest.warns(RuntimeWarning, match="Optional branch checkpoint missing for clip_only"):
        result = detector.predict(tiny_png)

    assert result["ai_prob"] == pytest.approx(torch.sigmoid(torch.tensor(-1.0)).item())
    assert result["pred_label"] == "Real"
    assert result["confidence"] == "medium"
    assert result["clip_score"] is None
    assert result["frequency_score"] == pytest.approx(torch.sigmoid(torch.tensor(0.5)).item())
    assert result["fusion_score"] == result["ai_prob"]
    assert Path(str(result["spectrum_path"])).is_file()
    assert Path(str(result["radial_spectrum_path"])).is_file()


def test_clip_detector_uses_clip_preprocess_for_uploaded_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _write_mlp_checkpoint(config, "clip_only", feature_type="clip", input_dim=4, bias=0.25)
    image_path = tmp_path / "upload.png"
    Image.new("RGB", (333, 217), color=(30, 80, 130)).save(image_path)

    fake_model = _FakeClipModel()
    monkeypatch.setattr("src.inference.detector_service.load_clip_model_and_preprocess", lambda _config, device: (fake_model, _fake_clip_preprocess))

    result = DetectorService(config_path=config_path, model_name="clip_only").predict(image_path)

    assert result["ai_prob"] == pytest.approx(torch.sigmoid(torch.tensor(0.25)).item())
    assert fake_model.seen_shape == (1, 3, 224, 224)



def _write_config(tmp_path: Path) -> Path:
    config = {
        "project": {"seed": 42, "device": "cpu"},
        "paths": {
            "checkpoint_dir": (tmp_path / "checkpoints").as_posix(),
            "figure_dir": (tmp_path / "figures").as_posix(),
        },
        "data": {"image_size": 24, "batch_size": 2, "num_workers": 0},
        "clip": {"model_name": "ViT-B-32", "pretrained": "openai", "output_dim": 4, "freeze": True, "normalize_feature": False},
        "frequency": {"method": "dct", "image_size": 24, "radial_bins": 10, "log_scale": True, "normalize_feature": True},
        "classifier": {"type": "mlp", "hidden_dim": 8, "dropout": 0.0, "num_classes": 1},
        "eval": {"threshold": 0.5},
        "demo": {"confidence": {"high_margin": 0.25, "medium_margin": 0.10}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _write_mlp_checkpoint(config: dict[str, Any], stem: str, *, feature_type: str, input_dim: int, bias: float) -> None:
    model = MLPClassifier(input_dim=input_dim, hidden_dim=8, dropout=0.0)
    _zero_model_with_bias(model, bias)
    save_checkpoint(
        Path(str(config["paths"]["checkpoint_dir"])) / f"{stem}.pt",
        model_state_dict=model.state_dict(),
        model_name="MLPClassifier",
        input_dim=input_dim,
        hidden_dim=8,
        threshold=0.5,
        feature_type=feature_type,
        config_snapshot=config,
    )


def _write_fusion_checkpoint(config: dict[str, Any], *, bias: float) -> None:
    model = FusionClassifier(clip_dim=4, freq_dim=10, hidden_dim=8, dropout=0.0)
    _zero_model_with_bias(model, bias)
    save_checkpoint(
        Path(str(config["paths"]["checkpoint_dir"])) / "fusion.pt",
        model_state_dict=model.state_dict(),
        model_name="FusionClassifier",
        input_dim=14,
        hidden_dim=8,
        threshold=0.5,
        feature_type="fusion",
        config_snapshot=config,
    )


def _zero_model_with_bias(model: torch.nn.Module, bias: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        last_bias = list(model.parameters())[-1]
        last_bias.fill_(bias)


class _FakeClipModel:
    def __init__(self) -> None:
        self.seen_shape: tuple[int, ...] | None = None

    def to(self, _device: object) -> "_FakeClipModel":
        return self

    def eval(self) -> None:
        return None

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(images.shape)
        return torch.ones((int(images.shape[0]), 4), dtype=torch.float32, device=images.device)


def _fake_clip_preprocess(_image: Image.Image) -> torch.Tensor:
    return torch.ones((3, 224, 224), dtype=torch.float32)


def test_demo_detector_service_loads_current_frequency_checkpoint(tmp_path: Path) -> None:
    import yaml
    from src.demo.detector_service import DetectorService
    from src.models.checkpoint import save_checkpoint
    from src.models.mlp_classifier import MLPClassifier

    checkpoint_dir = tmp_path / "checkpoints"
    model = MLPClassifier(input_dim=140, hidden_dim=4, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.05)
    save_checkpoint(
        checkpoint_dir / "frequency_only.pt",
        model_state_dict=model.state_dict(),
        model_name="MLPClassifier",
        input_dim=140,
        hidden_dim=4,
        threshold=0.5,
        feature_type="frequency",
        config_snapshot={},
    )
    config_path = tmp_path / "fusion.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"checkpoint_dir": str(checkpoint_dir / "frequency_only")}, "data": {"image_size": 32}}), encoding="utf-8")

    result = DetectorService(str(config_path)).predict(Image.new("RGB", (32, 32), (10, 20, 30)))

    assert result["status"] == "ok"
    assert result["frequency_prob"] != 0.5
