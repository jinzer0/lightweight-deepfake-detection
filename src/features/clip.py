from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.image_io import load_rgb_image

DEFAULT_CLIP_MODEL_ID = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CLIP_FEATURE_DIM = 512
CLIP_FEATURE_DTYPE = np.float32
CLIP_NORMALIZATION = "clip_l2"


class ClipDependencyError(ImportError):
    pass


class CudaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipFeatureConfig:
    model_id: str = DEFAULT_CLIP_MODEL_ID
    batch_size: int = 32
    device: str = "auto"
    normalize: bool = True

    def as_dict(self) -> dict[str, bool | int | str]:
        return {
            "model_id": str(self.model_id),
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "normalize": bool(self.normalize),
            "feature_dim": CLIP_FEATURE_DIM,
            "normalization": CLIP_NORMALIZATION if self.normalize else "none",
        }


def resolve_clip_device(device: str) -> str:
    torch = _require_torch()
    requested = device.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise CudaUnavailableError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false; refusing to fall back to CPU")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return requested


def preprocess_hash(preprocess: Any) -> str:
    return hashlib.sha256(repr(preprocess).encode("utf-8")).hexdigest()


def load_clip_model(*, model_id: str = DEFAULT_CLIP_MODEL_ID, device: str = "auto") -> tuple[Any, Any, str, str]:
    open_clip = _require_open_clip()
    resolved_device = resolve_clip_device(device)
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess, resolved_device, preprocess_hash(preprocess)


def extract_clip_feature_batch(
    image_paths: list[str | Path],
    *,
    model: Any,
    preprocess: Any,
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    torch = _require_torch()
    tensors = []
    for image_path in image_paths:
        image = load_rgb_image(image_path)
        tensors.append(preprocess(image))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        encoded = model.encode_image(batch, normalize=normalize)
    features = encoded.detach().cpu().numpy().astype(CLIP_FEATURE_DTYPE, copy=False)
    if features.ndim != 2:
        raise ValueError(f"CLIP features must be 2D, got shape {features.shape}")
    if int(features.shape[1]) != CLIP_FEATURE_DIM:
        raise ValueError(f"CLIP feature dimension must be {CLIP_FEATURE_DIM}, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("CLIP features contain non-finite values")
    if normalize:
        norms = np.linalg.norm(features, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("CLIP features are not L2-normalized")
    return features


def extract_clip_features(
    image_paths: list[str | Path],
    *,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    batch_size: int = 32,
    device: str = "auto",
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model, preprocess, resolved_device, resolved_preprocess_hash = load_clip_model(model_id=model_id, device=device)
    feature_batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        feature_batches.append(
            extract_clip_feature_batch(
                image_paths[start : start + batch_size],
                model=model,
                preprocess=preprocess,
                device=resolved_device,
                normalize=normalize,
            )
        )
    features = np.vstack(feature_batches).astype(CLIP_FEATURE_DTYPE, copy=False)
    return features, {"device": resolved_device, "preprocess_hash": resolved_preprocess_hash}


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on active environment
        raise ClipDependencyError("torch is required for CLIP feature extraction; install dependencies with `pip install -r requirements.txt`.") from exc
    return torch


def _require_open_clip() -> Any:
    try:
        import open_clip
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on active environment
        raise ClipDependencyError("open_clip_torch is required for CLIP feature extraction; install dependencies with `pip install -r requirements.txt`.") from exc
    return open_clip
