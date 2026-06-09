from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportAttributeAccessIssue=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from src.data.manifest import LABEL_TO_CLASS
from src.features.frequency import DEFAULT_FFT_EPSILON, DEFAULT_RADIAL_BINS, extract_frequency_features

LIMITATION_WARNING = (
    "This detector is an experimental model trained on limited benchmark data.\n"
    "It should not be used as definitive evidence that an image is real or AI-generated."
)
SUPPORTED_MODES = {"frequency_only", "clip_only", "fusion"}
REQUIRED_ARTIFACTS = ("config.yaml", "model.joblib", "scaler.joblib")


class PredictorArtifactError(ValueError):
    pass


class UnsupportedPredictorArtifactError(PredictorArtifactError):
    pass


@dataclass(frozen=True)
class PredictionResult:
    prob_fake: float
    score: float
    threshold: float
    artifact_threshold: float
    effective_threshold: float
    pred_label: int
    label_text: str
    warnings: list[str]
    limitations: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prob_fake": self.prob_fake,
            "score": self.score,
            "threshold": self.threshold,
            "artifact_threshold": self.artifact_threshold,
            "effective_threshold": self.effective_threshold,
            "pred_label": self.pred_label,
            "label_text": self.label_text,
            "warnings": list(self.warnings),
            "limitations": dict(self.limitations),
        }


class ImagePredictor:
    experiment_dir: Path
    config: dict[str, Any]
    model: Any
    transformers: dict[str, Any]
    mode: str
    artifact_threshold: float

    def __init__(self, *, experiment_dir: Path, config: dict[str, Any], model: Any, transformers: dict[str, Any]) -> None:
        self.experiment_dir = experiment_dir
        self.config = config
        self.model = model
        self.transformers = transformers
        self.mode = str(config["mode"])
        self.artifact_threshold = float(config.get("threshold", 0.5))

    @classmethod
    def from_experiment_dir(cls, path: str | Path) -> "ImagePredictor":
        experiment_dir = Path(path)
        if not experiment_dir.exists():
            raise PredictorArtifactError(f"experiment directory does not exist: {experiment_dir}")
        if not experiment_dir.is_dir():
            raise PredictorArtifactError(f"experiment path must be a directory: {experiment_dir}")

        missing = [file_name for file_name in REQUIRED_ARTIFACTS if not (experiment_dir / file_name).is_file()]
        if missing:
            raise PredictorArtifactError(f"missing required artifact(s): {', '.join(missing)}")

        config = _load_config(experiment_dir / "config.yaml")
        _validate_probability_policy(config)
        mode = _validate_mode(config)
        _validate_branch_metadata(config, mode)

        if mode != "frequency_only":
            _raise_unsupported_clip_or_fusion(mode, config)

        model = joblib.load(experiment_dir / "model.joblib")
        transformers = joblib.load(experiment_dir / "scaler.joblib")
        if not isinstance(transformers, dict):
            raise PredictorArtifactError("scaler.joblib must contain a transformer dictionary")
        _validate_frequency_predictor(model, transformers, config)
        return cls(experiment_dir=experiment_dir, config=config, model=model, transformers=transformers)

    def predict(self, image_path_or_pil: Any, threshold: float | None = None) -> PredictionResult:
        if self.mode != "frequency_only":
            _raise_unsupported_clip_or_fusion(self.mode, self.config)

        effective_threshold = self.artifact_threshold if threshold is None else float(threshold)
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {effective_threshold}")

        raw_features = _extract_frequency_vector(image_path_or_pil, self.config)
        scaler = self.transformers["frequency_scaler"]
        transformed = np.asarray(scaler.transform(raw_features.reshape(1, -1)), dtype=np.float32)
        if transformed.shape[0] != 1 or not np.isfinite(transformed).all():
            raise PredictorArtifactError("frequency scaler produced invalid transformed features")

        prob_fake = _predict_prob_fake(self.model, transformed)
        score = _predict_score(self.model, transformed)
        pred_label = int(prob_fake >= effective_threshold)
        label_text = LABEL_TO_CLASS.get(pred_label, str(pred_label))
        return PredictionResult(
            prob_fake=prob_fake,
            score=score,
            threshold=effective_threshold,
            artifact_threshold=self.artifact_threshold,
            effective_threshold=effective_threshold,
            pred_label=pred_label,
            label_text=label_text,
            warnings=[LIMITATION_WARNING],
            limitations={
                "experimental_warning": LIMITATION_WARNING,
                "mode": self.mode,
                "probability_supported": True,
                "decision_score_only": False,
            },
        )


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise PredictorArtifactError("config.yaml must contain a mapping")
    return payload


def _validate_probability_policy(config: dict[str, Any]) -> None:
    classifier_value = config.get("classifier")
    classifier: dict[str, Any] = classifier_value if isinstance(classifier_value, dict) else {}
    probability_supported = bool(config.get("probability_supported", classifier.get("probability_supported", False)))
    decision_score_only = bool(config.get("decision_score_only", classifier.get("decision_score_only", False)))
    streamlit_eligible = bool(config.get("streamlit_probability_eligible", classifier.get("streamlit_probability_eligible", False)))
    if not probability_supported or decision_score_only or not streamlit_eligible:
        classifier_key = classifier.get("key") or classifier.get("type") or "unknown"
        raise UnsupportedPredictorArtifactError(
            "artifact is not eligible for probability prediction "
            + f"(classifier={classifier_key}, probability_supported={probability_supported}, "
            + f"decision_score_only={decision_score_only}, streamlit_probability_eligible={streamlit_eligible}); "
            + "use a LogisticRegression probability artifact instead of a score-only Linear SVM artifact"
        )


def _validate_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("mode", ""))
    if mode not in SUPPORTED_MODES:
        raise UnsupportedPredictorArtifactError(f"unsupported feature mode for inference: {mode!r}")
    return mode


def _validate_branch_metadata(config: dict[str, Any], mode: str) -> None:
    feature = config.get("feature")
    if not isinstance(feature, dict):
        raise PredictorArtifactError("config.yaml is missing feature metadata")
    branches = feature.get("branches")
    if not isinstance(branches, dict):
        raise PredictorArtifactError("config.yaml is missing feature.branches metadata")

    required = ["frequency"] if mode == "frequency_only" else ["clip"] if mode == "clip_only" else ["frequency", "clip"]
    for branch_name in required:
        branch = branches.get(branch_name)
        if not isinstance(branch, dict):
            raise PredictorArtifactError(f"config.yaml is missing feature.branches.{branch_name} metadata")
        if str(branch.get("feature_type", "")) != branch_name:
            raise PredictorArtifactError(f"feature.branches.{branch_name}.feature_type must be {branch_name}")
        if "feature_dim" not in branch:
            raise PredictorArtifactError(f"feature.branches.{branch_name}.feature_dim is required")


def _validate_frequency_predictor(model: Any, transformers: dict[str, Any], config: dict[str, Any]) -> None:
    if "frequency_scaler" not in transformers:
        raise PredictorArtifactError("scaler.joblib is missing required transformer key: frequency_scaler")
    scaler = transformers["frequency_scaler"]
    if not hasattr(scaler, "transform"):
        raise PredictorArtifactError("frequency_scaler must expose transform(); prediction must not fit a scaler")
    if not hasattr(model, "predict_proba"):
        raise UnsupportedPredictorArtifactError("model.joblib does not expose predict_proba(); use a probability-capable LogisticRegression artifact")
    classes = np.asarray(getattr(model, "classes_", []), dtype=np.int64)
    if np.count_nonzero(classes == 1) != 1:
        raise PredictorArtifactError(f"model classes must contain fake label 1 exactly once, got {classes.tolist()}")
    branch = config["feature"]["branches"]["frequency"]
    if int(branch.get("feature_dim")) != 220:
        raise PredictorArtifactError(f"frequency feature_dim must be 220, got {branch.get('feature_dim')}")


def _extract_frequency_vector(image: Any, config: dict[str, Any]) -> np.ndarray:
    metadata = config["feature"]["branches"]["frequency"].get("metadata", {})
    image_size = int(metadata.get("image_size", 224))
    radial_bins = int(metadata.get("radial_bins", DEFAULT_RADIAL_BINS))
    fft_epsilon = float(metadata.get("fft_epsilon", DEFAULT_FFT_EPSILON))
    features = np.asarray(
        extract_frequency_features(image, image_size=image_size, radial_bins=radial_bins, fft_epsilon=fft_epsilon),
        dtype=np.float32,
    )
    if features.shape != (220,):
        raise PredictorArtifactError(f"frequency extraction must produce 220 features, got shape {features.shape}")
    return features


def _predict_prob_fake(model: Any, features: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probabilities.shape[0] != 1:
        raise PredictorArtifactError(f"predict_proba must return one row, got shape {probabilities.shape}")
    classes = np.asarray(model.classes_, dtype=np.int64)
    fake_indices = np.flatnonzero(classes == 1)
    if fake_indices.size != 1:
        raise PredictorArtifactError(f"model classes must contain fake label 1 exactly once, got {classes.tolist()}")
    prob_fake = float(probabilities[0, int(fake_indices[0])])
    if not 0.0 <= prob_fake <= 1.0:
        raise PredictorArtifactError(f"model returned prob_fake outside [0, 1]: {prob_fake}")
    return prob_fake


def _predict_score(model: Any, features: np.ndarray) -> float:
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features), dtype=np.float64)
        if decision.ndim == 1:
            return float(decision[0])
        classes = np.asarray(model.classes_, dtype=np.int64)
        fake_indices = np.flatnonzero(classes == 1)
        if fake_indices.size == 1:
            return float(decision[0, int(fake_indices[0])])
    return _probability_logit(_predict_prob_fake(model, features))


def _probability_logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(np.log(clipped / (1.0 - clipped)))


def _raise_unsupported_clip_or_fusion(mode: str, config: dict[str, Any]) -> None:
    branches = config.get("feature", {}).get("branches", {}) if isinstance(config.get("feature"), dict) else {}
    clip_branch = branches.get("clip", {}) if isinstance(branches, dict) else {}
    clip_metadata = clip_branch.get("metadata", {}) if isinstance(clip_branch, dict) else {}
    missing = [key for key in ("model_name", "preprocess_hash", "device") if key not in clip_metadata]
    if missing:
        detail = f"missing CLIP metadata: {', '.join(missing)}"
    else:
        detail = "saved config does not include enough reloadable openCLIP preprocessing metadata to guarantee cache-compatible live features"
    raise UnsupportedPredictorArtifactError(
        f"{mode} artifacts are not supported by ImagePredictor probability inference yet: {detail}; "
        + "provide a frequency_only LogisticRegression artifact or add a reloadable CLIP preprocessing contract"
    )
