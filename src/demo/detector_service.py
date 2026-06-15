from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
import yaml
from PIL import Image

from src.features.frequency_features import extract_frequency_feature_dict, extract_frequency_features
from src.models.checkpoint import CheckpointError, load_checkpoint
from src.models.frequency_classifier import FrequencyClassifier
from src.models.mlp_classifier import MLPClassifier


def neutral_probability() -> float:
    return 0.5


class DetectorService:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}
        demo_cfg = self.config.get("demo", {}) if isinstance(self.config, dict) else {}
        eval_cfg = self.config.get("eval", {}) if isinstance(self.config, dict) else {}
        self.threshold = float(demo_cfg.get("threshold", eval_cfg.get("threshold", 0.5)))
        self.high_bounds = tuple(demo_cfg.get("high_confidence", [0.2, 0.8]))
        self.medium_bounds = tuple(demo_cfg.get("medium_confidence", [0.35, 0.65]))
        self.frequency_model = None
        self.frequency_reason = "frequency checkpoint missing"
        self._try_load_frequency_model()

    def predict(self, image: Any) -> dict[str, Any]:
        """Return branch scores, final label, confidence, and frequency explanations."""
        if image is None:
            return self._response(neutral_probability(), neutral_probability(), neutral_probability(), neutral_probability(), 0.0, [])
        pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
        feature_info = extract_frequency_feature_dict(pil, image_size=int(self.config.get("data", {}).get("image_size", 512)))
        freq_prob = self._frequency_probability(feature_info["feature"])
        clip_prob = neutral_probability()
        resnet_prob = neutral_probability()
        fusion_prob = float(np.mean([freq_prob, clip_prob, resnet_prob])) if self.frequency_model is not None else neutral_probability()
        return self._response(fusion_prob, clip_prob, freq_prob, resnet_prob, float(feature_info["high_frequency_ratio"]), [float(x) for x in feature_info["fft_radial_spectrum"]], plots={})

    def _try_load_frequency_model(self) -> None:
        checkpoint = self._frequency_checkpoint_path()
        if checkpoint is None:
            return
        try:
            if checkpoint.name == "frequency_only.pt":
                payload = load_checkpoint(checkpoint, expected_feature_type="frequency")
                input_dim = cast(int, payload["input_dim"])
                hidden_dim = cast(int, payload["hidden_dim"])
                model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
                state = cast(Mapping[str, Any], payload["model_state_dict"])
            else:
                raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                payload = cast(Mapping[str, Any], raw_payload)
                state = cast(Mapping[str, Any], payload["model_state_dict"])
                first_weight = next(value for key, value in state.items() if key.endswith("weight") and getattr(value, "ndim", 0) == 2)
                model = FrequencyClassifier(input_dim=int(first_weight.shape[1]))
            model.load_state_dict(state)
            model.eval()
            self.frequency_model = model
            self.frequency_reason = "ok"
        except (RuntimeError, KeyError, StopIteration, ValueError, CheckpointError) as exc:
            self.frequency_reason = f"frequency checkpoint unavailable: {exc}"

    def _frequency_checkpoint_path(self) -> Path | None:
        paths = self.config.get("paths", {}) if isinstance(self.config, dict) else {}
        candidates = []
        checkpoint_dir = Path(paths.get("checkpoint_dir", "artifacts/checkpoints/frequency_only"))
        if checkpoint_dir.suffix == ".pt":
            candidates.append(checkpoint_dir)
        else:
            candidates.extend([checkpoint_dir / "best_checkpoint.pt", checkpoint_dir.parent / "frequency_only.pt"])
        candidates.append(Path("artifacts/checkpoints/frequency_only.pt"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        self.frequency_reason = f"frequency checkpoint missing: {candidates[0]}"
        return None

    def _frequency_probability(self, feature: np.ndarray) -> float:
        if self.frequency_model is None:
            return neutral_probability()
        with torch.inference_mode():
            return float(torch.sigmoid(self.frequency_model(torch.from_numpy(feature[None, :]).float())).item())

    def _confidence(self, prob: float) -> str:
        if prob >= float(self.high_bounds[1]) or prob <= float(self.high_bounds[0]):
            return "높음"
        if prob >= float(self.medium_bounds[1]) or prob <= float(self.medium_bounds[0]):
            return "중간"
        return "낮음"

    def _response(self, fusion_prob: float, clip_prob: float, frequency_prob: float, resnet50_prob: float, high_frequency_ratio: float, radial_spectrum: list[float], plots: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "fusion_prob": fusion_prob,
            "clip_prob": clip_prob,
            "frequency_prob": frequency_prob,
            "resnet50_prob": resnet50_prob,
            "final_label": "AI-generated" if fusion_prob >= self.threshold else "Real",
            "confidence": self._confidence(fusion_prob),
            "high_frequency_ratio": high_frequency_ratio,
            "radial_spectrum": radial_spectrum,
            "plots": dict(plots or {}),
            "status": "ok" if self.frequency_model is not None else "not run",
            "reason": self.frequency_reason if self.frequency_model is None else "frequency checkpoint loaded; other branches neutral unless artifacts are integrated",
        }


DemoDetectorService = DetectorService
