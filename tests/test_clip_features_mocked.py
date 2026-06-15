from __future__ import annotations


import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import clip_features
from src.features.clip_features import ClipModelLoadError, extract_clip_features, l2_normalize, load_clip_model, load_clip_model_and_preprocess


def test_load_clip_model_uses_open_clip_config_and_freezes_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingClipModel()
    calls: list[dict[str, object]] = []

    def create_model_and_transforms(model_name: str, *, pretrained: str) -> tuple[RecordingClipModel, object, object]:
        calls.append({"model_name": model_name, "pretrained": pretrained})
        return model, object(), object()

    fake_open_clip = types.SimpleNamespace(create_model_and_transforms=create_model_and_transforms)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    loaded = load_clip_model(_config(freeze=True), device="cpu")

    assert loaded is model
    assert calls == [{"model_name": "ViT-B-32", "pretrained": "openai"}]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert all(not parameter.requires_grad for parameter in model.parameters())



def test_load_clip_model_and_preprocess_uses_hf_first(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingClipModel()
    preprocess = object()
    calls: list[str] = []

    def create_model_from_pretrained(model_id: str) -> tuple[RecordingClipModel, object]:
        calls.append(model_id)
        return model, preprocess

    fake_open_clip = types.SimpleNamespace(create_model_from_pretrained=create_model_from_pretrained)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    loaded_model, loaded_preprocess = load_clip_model_and_preprocess({"clip": {"hf_hub_model": "hf-hub:test/model"}}, device="cpu")

    assert loaded_model is model
    assert loaded_preprocess is preprocess
    assert calls == ["hf-hub:test/model"]


def test_load_clip_model_keeps_trainable_parameters_when_freeze_false(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingClipModel()

    def create_model_and_transforms(model_name: str, *, pretrained: str) -> tuple[RecordingClipModel, object, object]:
        return model, object(), object()

    fake_open_clip = types.SimpleNamespace(create_model_and_transforms=create_model_and_transforms)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    loaded = load_clip_model(_config(freeze=False), device="cpu")

    assert loaded is model
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_extract_clip_features_returns_cpu_numpy_labels_and_metadata() -> None:
    dataset = TensorMetadataDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = RecordingClipModel()

    features, labels, metadata = extract_clip_features(model, dataloader, device="cpu", normalize=True)

    assert features.dtype == np.float32
    assert labels.dtype == np.int64
    assert features.shape == (3, 4)
    assert labels.tolist() == [0, 1, 0]
    assert np.allclose(np.linalg.norm(features[:2], axis=1), 1.0)
    assert np.array_equal(features[2], np.zeros(4, dtype=np.float32))
    assert list(metadata["image_id"]) == ["img-0", "img-1", "img-2"]
    assert list(metadata["filepath"]) == ["/tmp/img-0.png", "/tmp/img-1.png", "/tmp/img-2.png"]
    assert list(metadata["label"]) == [0, 1, 0]
    assert {"class_name", "dataset", "generator", "split"}.issubset(metadata.columns)
    assert model.eval_calls == 1
    assert len(model.seen_batches) == 2


def test_extract_clip_features_does_not_apply_open_clip_preprocess() -> None:
    dataloader = DataLoader(TensorMetadataDataset(), batch_size=3, shuffle=False)
    model = RecordingClipModel()

    _ = extract_clip_features(model, dataloader, device="cpu", normalize=False)

    assert torch.equal(model.seen_batches[0], torch.stack([TensorMetadataDataset.image_for(index) for index in range(3)]))


def test_l2_normalize_handles_zero_and_near_zero_rows() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0], [1.0e-14, 0.0]], dtype=np.float32)

    normalized = l2_normalize(features)

    assert normalized.dtype == np.float32
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.array_equal(normalized[1], [0.0, 0.0])
    assert np.array_equal(normalized[2], features[2])


def test_load_failure_message_names_package_model_pretrained_and_offline_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def create_model_and_transforms(model_name: str, *, pretrained: str) -> tuple[object, object, object]:
        raise RuntimeError("network disabled")

    fake_open_clip = types.SimpleNamespace(create_model_and_transforms=create_model_and_transforms)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    with pytest.raises(ClipModelLoadError) as error:
        load_clip_model(_config(), device="cpu")

    message = str(error.value)
    assert "open_clip_torch" in message
    assert "ViT-B-32" in message
    assert "openai" in message
    assert "offline optional-smoke" in message
    assert "network disabled" in message


def _config(*, freeze: bool = True) -> dict[str, dict[str, object]]:
    return {"clip": {"model_name": "ViT-B-32", "pretrained": "openai", "freeze": freeze}}


class RecordingClipModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.ones(1))
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.seen_batches: list[torch.Tensor] = []

    def to(self, *args: Any, **kwargs: Any) -> "RecordingClipModel":
        device = args[0] if args else kwargs.get("device", "")
        self.to_calls.append(str(device))
        return self

    def eval(self) -> "RecordingClipModel":
        self.eval_calls += 1
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        self.seen_batches.append(images.detach().cpu().clone())
        return images.flatten(start_dim=1)[:, :4]


class TensorMetadataDataset(Dataset[tuple[torch.Tensor, int, dict[str, str]]]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        label = index % 2
        return self.image_for(index), label, {
            "image_id": f"img-{index}",
            "filepath": f"/tmp/img-{index}.png",
            "class_name": "fake" if label else "real",
            "dataset": "mock",
            "generator": "unit-test",
            "split": "train",
        }

    @staticmethod
    def image_for(index: int) -> torch.Tensor:
        if index == 2:
            return torch.zeros((1, 2, 2), dtype=torch.float32)
        return torch.full((1, 2, 2), float(index + 1), dtype=torch.float32)


def test_import_safe_without_open_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", None)

    assert clip_features.l2_normalize(np.zeros((1, 2), dtype=np.float32)).shape == (1, 2)


def test_mocked_clip_evidence_directory_can_be_written() -> None:
    evidence_dir = Path(".omo/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    assert evidence_dir.is_dir()
