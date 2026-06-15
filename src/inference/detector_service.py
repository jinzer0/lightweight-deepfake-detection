from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportOptionalMemberAccess=false, reportArgumentType=false

import re
import uuid
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from src.data.transforms import get_eval_transform
from src.features.clip_features import l2_normalize, load_clip_model
from src.features.frequency_features import extract_frequency_feature
from src.models.checkpoint import CheckpointError, load_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.models.mlp_classifier import MLPClassifier
from src.utils.config import load_config, resolve_device
from src.utils.image_io import CorruptImageError, load_rgb_image
from src.visualization.radial_spectrum import save_radial_spectrum_plot
from src.visualization.spectrum import save_spectrum_image


MODEL_SPECS = {
    "frequency_only": {"checkpoint": "frequency_only.pt", "feature_type": "frequency"},
    "clip_only": {"checkpoint": "clip_only.pt", "feature_type": "clip"},
    "fusion": {"checkpoint": "fusion.pt", "feature_type": "fusion"},
}
RESULT_KEYS = (
    "ai_prob",
    "pred_label",
    "confidence",
    "clip_score",
    "frequency_score",
    "fusion_score",
    "spectrum_path",
    "radial_spectrum_path",
)


class DetectorServiceError(ValueError):
    pass


class DetectorService:
    def __init__(self, config_path: str | Path = "configs/default.yaml", model_name: str = "fusion") -> None:
        if model_name not in MODEL_SPECS:
            raise DetectorServiceError(f"model_name must be one of {sorted(MODEL_SPECS)}, got {model_name!r}")
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path.as_posix())
        self.model_name = model_name
        self.device = torch.device(resolve_device(dict(self.config)))
        self.checkpoint_dir = _path_from_config(self.config, "checkpoint_dir", "artifacts/checkpoints")
        self.figure_dir = _path_from_config(self.config, "figure_dir", "artifacts/figures")
        self.model, self.threshold = self._load_selected_model(model_name)
        self._clip_model: Any | None = None

    def predict(self, image_path: str | Path) -> dict[str, float | str | None]:
        path = Path(image_path)
        image = _load_image(path)
        frequency_feature = self._extract_frequency_feature(image)
        spectrum_path, radial_spectrum_path = self._write_visualizations(path, image, frequency_feature)

        clip_feature: np.ndarray | None = None
        if self.model_name in {"clip_only", "fusion"}:
            clip_feature = self._extract_clip_feature(image)

        selected_feature = self._selected_feature(clip_feature, frequency_feature)
        ai_prob = self._predict_probability(self.model, selected_feature)
        result = {
            "ai_prob": ai_prob,
            "pred_label": "AI" if ai_prob >= self.threshold else "Real",
            "confidence": self._confidence(ai_prob),
            "clip_score": None,
            "frequency_score": None,
            "fusion_score": None,
            "spectrum_path": spectrum_path,
            "radial_spectrum_path": radial_spectrum_path,
        }

        if self.model_name == "clip_only":
            result["clip_score"] = ai_prob
        elif self.model_name == "frequency_only":
            result["frequency_score"] = ai_prob
        else:
            result["fusion_score"] = ai_prob
            result["clip_score"] = self._optional_branch_score("clip_only", clip_feature, frequency_feature)
            result["frequency_score"] = self._optional_branch_score("frequency_only", clip_feature, frequency_feature)

        return {key: result[key] for key in RESULT_KEYS}

    def _load_selected_model(self, model_name: str) -> tuple[nn.Module, float]:
        spec = MODEL_SPECS[model_name]
        checkpoint_path = self.checkpoint_dir / str(spec["checkpoint"])
        checkpoint = load_checkpoint(checkpoint_path, expected_feature_type=str(spec["feature_type"]))
        model = self._build_model(checkpoint)
        model.to(self.device)
        model.eval()
        return model, _float_checkpoint_value(checkpoint["threshold"], "threshold")

    def _build_model(self, checkpoint: Mapping[str, object]) -> nn.Module:
        input_dim = _int_checkpoint_value(checkpoint["input_dim"], "input_dim")
        hidden_dim = _int_checkpoint_value(checkpoint["hidden_dim"], "hidden_dim")
        model_name = str(checkpoint["model_name"])
        feature_type = str(checkpoint["feature_type"])

        if feature_type == "fusion":
            if model_name != "FusionClassifier":
                raise CheckpointError(f"fusion checkpoint model_name must be FusionClassifier, got {model_name!r}")
            frequency_dim = _frequency_dim(self.config)
            clip_dim = input_dim - frequency_dim
            if clip_dim <= 0:
                raise CheckpointError(f"fusion checkpoint input_dim {input_dim} cannot contain frequency_dim {frequency_dim}")
            model: nn.Module = FusionClassifier(clip_dim=clip_dim, freq_dim=frequency_dim, hidden_dim=hidden_dim, dropout=0.0)
        else:
            if model_name != "MLPClassifier":
                raise CheckpointError(f"{feature_type} checkpoint model_name must be MLPClassifier, got {model_name!r}")
            model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)

        state = cast(Mapping[str, torch.Tensor], checkpoint["model_state_dict"])
        model.load_state_dict(state)
        return model

    def _extract_frequency_feature(self, image: Any) -> np.ndarray:
        feature = np.asarray(extract_frequency_feature(image, self.config), dtype=np.float32)
        expected_dim = _frequency_dim(self.config)
        if feature.shape != (expected_dim,):
            raise DetectorServiceError(f"frequency feature shape must be ({expected_dim},), got {feature.shape}")
        return feature

    def _extract_clip_feature(self, image: Any) -> np.ndarray:
        if self._clip_model is None:
            self._clip_model = load_clip_model(self.config, device=self.device)
        transform = get_eval_transform(_image_size(self.config))
        tensor = transform(image).unsqueeze(0).to(self.device)
        self._clip_model.eval()
        with torch.inference_mode():
            encoded = self._clip_model.encode_image(tensor)
        feature = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        if _clip_normalize(self.config):
            feature = l2_normalize(feature)
        if feature.ndim != 2 or feature.shape[0] != 1:
            raise DetectorServiceError(f"CLIP extractor must return one feature row, got shape {feature.shape}")
        return feature[0].astype(np.float32, copy=False)

    def _selected_feature(self, clip_feature: np.ndarray | None, frequency_feature: np.ndarray) -> np.ndarray:
        if self.model_name == "frequency_only":
            return frequency_feature
        if self.model_name == "clip_only":
            if clip_feature is None:
                raise DetectorServiceError("clip_only prediction requires a CLIP feature")
            return clip_feature
        if clip_feature is None:
            raise DetectorServiceError("fusion prediction requires a CLIP feature")
        return np.concatenate([clip_feature, frequency_feature], axis=0).astype(np.float32, copy=False)

    def _predict_probability(self, model: nn.Module, feature: np.ndarray) -> float:
        values = np.asarray(feature, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise DetectorServiceError(f"prediction feature must be a finite 1D array, got shape {values.shape}")
        tensor = torch.from_numpy(values.reshape(1, -1)).to(self.device)
        with torch.inference_mode():
            logits = model(tensor)
            probability = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
        if probability.shape != (1,):
            raise DetectorServiceError(f"model must return one logit, got probability shape {probability.shape}")
        ai_prob = float(probability[0])
        if not 0.0 <= ai_prob <= 1.0:
            raise DetectorServiceError(f"model returned fake probability outside [0, 1]: {ai_prob}")
        return ai_prob

    def _optional_branch_score(self, model_name: str, clip_feature: np.ndarray | None, frequency_feature: np.ndarray) -> float | None:
        spec = MODEL_SPECS[model_name]
        checkpoint_path = self.checkpoint_dir / str(spec["checkpoint"])
        if not checkpoint_path.is_file():
            warnings.warn(f"Optional branch checkpoint missing for {model_name}: {checkpoint_path}", RuntimeWarning, stacklevel=2)
            return None
        checkpoint = load_checkpoint(checkpoint_path, expected_feature_type=str(spec["feature_type"]))
        model = self._build_model(checkpoint)
        model.to(self.device)
        model.eval()
        feature = frequency_feature if model_name == "frequency_only" else clip_feature
        if feature is None:
            raise DetectorServiceError(f"{model_name} branch scoring requires a CLIP feature")
        return self._predict_probability(model, feature)

    def _write_visualizations(self, image_path: Path, image: Any, frequency_feature: np.ndarray) -> tuple[str, str]:
        stem = _safe_stem(image_path)
        token = uuid.uuid4().hex[:12]
        spectrum_path = self.figure_dir / f"{stem}_{token}_spectrum.png"
        radial_path = self.figure_dir / f"{stem}_{token}_radial.png"
        method = str(_frequency_settings(self.config).get("method", "dct"))
        saved_spectrum = save_spectrum_image(image, spectrum_path.as_posix(), method=method)
        saved_radial = save_radial_spectrum_plot(frequency_feature, radial_path.as_posix())
        return saved_spectrum, saved_radial

    def _confidence(self, ai_prob: float) -> str:
        settings = _confidence_settings(self.config)
        high_margin = float(settings.get("high_margin", 0.25))
        medium_margin = float(settings.get("medium_margin", 0.10))
        margin = abs(float(ai_prob) - 0.5)
        if margin >= high_margin:
            return "high"
        if margin >= medium_margin:
            return "medium"
        return "low"


def _load_image(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"image file not found: {path}")
    try:
        return load_rgb_image(path)
    except CorruptImageError:
        raise
    except Exception as exc:
        raise CorruptImageError(f"Failed to decode image: {path}") from exc


def _path_from_config(config: Mapping[str, object], key: str, default: str) -> Path:
    paths = config.get("paths")
    if isinstance(paths, Mapping):
        return Path(str(paths.get(key, default)))
    return Path(default)


def _frequency_settings(config: Mapping[str, object]) -> Mapping[str, object]:
    settings = config.get("frequency")
    return settings if isinstance(settings, Mapping) else {}


def _confidence_settings(config: Mapping[str, object]) -> Mapping[str, object]:
    demo = config.get("demo")
    if not isinstance(demo, Mapping):
        return {}
    confidence = demo.get("confidence")
    return confidence if isinstance(confidence, Mapping) else {}


def _frequency_dim(config: Mapping[str, object]) -> int:
    return int(_frequency_settings(config).get("radial_bins", 64))


def _image_size(config: Mapping[str, object]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "image_size" in data:
        return int(data["image_size"])
    return int(_frequency_settings(config).get("image_size", 512))


def _clip_normalize(config: Mapping[str, object]) -> bool:
    clip = config.get("clip")
    if isinstance(clip, Mapping):
        return bool(clip.get("normalize_feature", True))
    return True


def _safe_stem(image_path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._")
    return cleaned or "image"


def _int_checkpoint_value(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be an int")
    return value


def _float_checkpoint_value(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be numeric")
    return float(value)
