from __future__ import annotations

# pyright: reportMissingImports=false

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

CLIP_FEATURE_DTYPE = np.float32
DEFAULT_HF_MODEL = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
FALLBACK_MODEL = "ViT-L-14"
FALLBACK_PRETRAINED = "laion2B-s32B-b82K"


class ClipModelLoadError(RuntimeError):
    pass


def load_clip_model(config: Mapping[str, Any] | None = None, device: str | torch.device = "cpu") -> Any:
    model, _preprocess = load_clip_model_and_preprocess(config, device)
    return model


def load_clip_model_and_preprocess(config: Mapping[str, Any] | None = None, device: str | torch.device = "cpu") -> tuple[Any, Any]:
    cfg = dict(config or {})
    clip_cfg = dict(cfg.get("clip", cfg)) if isinstance(cfg.get("clip", cfg), Mapping) else cfg
    freeze = bool(clip_cfg.get("freeze", True))
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise ClipModelLoadError("open_clip_torch is required for CLIP feature extraction. If this is an offline optional-smoke run, ensure open_clip_torch and cached weights are available or skip live CLIP extraction. Install requirements.txt.") from exc

    hf_model = clip_cfg.get("hf_hub_model") or clip_cfg.get("model_id")
    fallback_model = str(clip_cfg.get("fallback_model_name", clip_cfg.get("model_name", FALLBACK_MODEL)))
    fallback_pretrained = str(clip_cfg.get("fallback_pretrained", clip_cfg.get("pretrained", FALLBACK_PRETRAINED)))
    if hf_model is not None:
        try:
            model, preprocess = open_clip.create_model_from_pretrained(str(hf_model))
        except Exception:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(fallback_model, pretrained=fallback_pretrained)
            except Exception as exc:
                raise ClipModelLoadError(f"open_clip_torch could not load {hf_model} or fallback {fallback_model}/{fallback_pretrained} during offline optional-smoke: {exc}") from exc
    else:
        model_name = str(clip_cfg.get("model_name", FALLBACK_MODEL))
        pretrained = str(clip_cfg.get("pretrained", FALLBACK_PRETRAINED))
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        except Exception as exc:
            raise ClipModelLoadError(f"open_clip_torch could not load {model_name}/{pretrained} during offline optional-smoke: {exc}") from exc
    model.to(device)
    model.eval()
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, preprocess

def extract_clip_features(model: Any, dataloader: Iterable[Any], device: str | torch.device, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    metadata_frames: list[pd.DataFrame] = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, labels, metadata = _unpack_batch(batch)
            encoded = model.encode_image(images.to(device))
            features = encoded.detach().cpu().numpy().astype(CLIP_FEATURE_DTYPE, copy=False)
            if normalize:
                features = l2_normalize(features)
            label_array = _labels_to_numpy(labels)
            feature_batches.append(features)
            label_batches.append(label_array)
            metadata_frames.append(_metadata_to_frame(metadata, label_array))
    if not feature_batches:
        return np.empty((0, 0), dtype=CLIP_FEATURE_DTYPE), np.empty((0,), dtype=np.int64), pd.DataFrame()
    return np.concatenate(feature_batches), np.concatenate(label_batches), pd.concat(metadata_frames, ignore_index=True)


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    out = features.astype(CLIP_FEATURE_DTYPE, copy=True)
    mask = norms[:, 0] >= eps
    out[mask] = out[mask] / norms[mask]
    return out.astype(CLIP_FEATURE_DTYPE, copy=False)


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, Any, Any]:
    if len(batch) == 3:
        return batch
    images, labels = batch
    return images, labels, {}


def _labels_to_numpy(labels: Any) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy().astype(np.int64)
    return np.asarray(labels, dtype=np.int64)


def _metadata_to_frame(metadata: Any, labels: np.ndarray) -> pd.DataFrame:
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif isinstance(metadata, dict):
        frame = pd.DataFrame(metadata)
    elif isinstance(metadata, list):
        frame = pd.DataFrame(metadata)
    else:
        frame = pd.DataFrame()
    if "label" not in frame:
        frame["label"] = labels
    return frame
