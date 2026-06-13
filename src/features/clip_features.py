from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch


CLIP_FEATURE_DTYPE = np.float32


class ClipModelLoadError(RuntimeError):
    pass


def load_clip_model(config: Mapping[str, Any], device: str | torch.device) -> Any:
    clip_config = _clip_config(config)
    model_name = str(clip_config["model_name"])
    pretrained = str(clip_config["pretrained"])

    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise ClipModelLoadError(_load_error_message(model_name=model_name, pretrained=pretrained, error=exc)) from exc

    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.to(device)
        model.eval()
        if bool(clip_config.get("freeze", True)):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        return model
    except Exception as exc:
        raise ClipModelLoadError(_load_error_message(model_name=model_name, pretrained=pretrained, error=exc)) from exc


def extract_clip_features(
    model: Any,
    dataloader: Iterable[Any],
    device: str | torch.device,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []

    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, labels, metadata = _unpack_batch(batch)
            images = images.to(device)
            encoded = model.encode_image(images)
            features = encoded.detach().cpu().numpy().astype(CLIP_FEATURE_DTYPE, copy=False)
            if normalize:
                features = l2_normalize(features)
            feature_batches.append(features)

            label_array = _labels_to_numpy(labels)
            label_batches.append(label_array)
            metadata_frames.append(_metadata_to_frame(metadata, label_array))

    if not feature_batches:
        return np.empty((0, 0), dtype=CLIP_FEATURE_DTYPE), np.empty((0,), dtype=np.int64), pd.DataFrame()

    features = np.concatenate(feature_batches, axis=0).astype(CLIP_FEATURE_DTYPE, copy=False)
    labels = np.concatenate(label_batches, axis=0).astype(np.int64, copy=False)
    metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame(index=range(len(labels)))
    if len(metadata_df) != int(features.shape[0]) or len(labels) != int(features.shape[0]):
        raise ValueError("CLIP feature, label, and metadata row counts must match")
    return features, labels, metadata_df


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(features, dtype=CLIP_FEATURE_DTYPE)
    if array.ndim != 2:
        raise ValueError(f"features must be a 2D array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms > eps, norms, 1.0).astype(CLIP_FEATURE_DTYPE, copy=False)
    normalized = array / safe_norms
    return normalized.astype(CLIP_FEATURE_DTYPE, copy=False)


def _clip_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "clip" not in config or not isinstance(config["clip"], Mapping):
        raise ValueError("config must contain a clip section")
    clip_config = config["clip"]
    missing = [key for key in ("model_name", "pretrained") if key not in clip_config]
    if missing:
        raise ValueError(f"config.clip missing required key(s): {', '.join(missing)}")
    return clip_config


def _load_error_message(*, model_name: str, pretrained: str, error: BaseException) -> str:
    return (
        "Failed to load frozen CLIP image encoder with open_clip_torch. "
        f"model_name={model_name!r}, pretrained={pretrained!r}. "
        "If this is an offline optional-smoke run, ensure open_clip_torch is installed and the requested weights are already cached, "
        "or skip the live CLIP smoke and rely on mocked/offline tests. "
        f"Original error: {type(error).__name__}: {error}"
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, Any, Mapping[str, Any] | None]:
    if not isinstance(batch, Sequence) or len(batch) not in {2, 3}:
        raise ValueError("dataloader batches must be (images, labels) or (images, labels, metadata)")
    images = batch[0]
    if not isinstance(images, torch.Tensor):
        raise TypeError("dataloader image batches must be torch.Tensor objects from project transforms")
    metadata = batch[2] if len(batch) == 3 else None
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("dataloader metadata must be a mapping produced by the default collate function")
    return images, batch[1], metadata


def _labels_to_numpy(labels: Any) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(labels, dtype=np.int64)


def _metadata_to_frame(metadata: Mapping[str, Any] | None, labels: np.ndarray) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame({"label": labels})

    rows: dict[str, list[Any]] = {}
    expected_len = len(labels)
    for key, values in metadata.items():
        rows[str(key)] = _metadata_column(values, expected_len)
    if "label" not in rows:
        rows["label"] = labels.tolist()
    return pd.DataFrame(rows)


def _metadata_column(values: Any, expected_len: int) -> list[Any]:
    if isinstance(values, torch.Tensor):
        column = values.detach().cpu().numpy().tolist()
    elif isinstance(values, np.ndarray):
        column = values.tolist()
    elif isinstance(values, (list, tuple)):
        column = list(values)
    else:
        column = [values]
    if len(column) != expected_len:
        raise ValueError("metadata column length must match batch labels")
    return column
