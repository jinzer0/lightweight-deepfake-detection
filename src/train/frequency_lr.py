from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportArgumentType=false, reportUnusedCallResult=false

import csv
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.svm import LinearSVC

from src.data.manifest import CLASS_TO_LABEL, read_manifest, validate_manifest_rows
from src.models.logistic_regression import CLASSIFIER_NAME
from src.train.features import assemble_features

FeatureMode = Literal["frequency_only", "clip_only", "fusion"]
ClassifierName = Literal["logistic_regression", "linear_svm"]
PREDICTION_COLUMNS = ["path", "sample_id", "label", "pred_label", "prob_fake", "score", "split"]


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    model_path: Path
    scaler_path: Path
    config_path: Path
    metrics_path: Path
    predictions_path: Path
    reload_max_abs_diff: float | None


def train_frequency_logistic_regression(
    *,
    manifest_path: str | Path,
    feature_cache_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    threshold: float = 0.5,
    max_iter: int = 1000,
    c_value: float = 1.0,
    verify_reload: bool = True,
    reload_tolerance: float = 1e-12,
    command: list[str] | None = None,
) -> TrainResult:
    return train_classifier(
        manifest_path=manifest_path,
        output_dir=output_dir,
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=feature_cache_path,
        seed=seed,
        threshold=threshold,
        max_iter=max_iter,
        c_value=c_value,
        verify_reload=verify_reload,
        reload_tolerance=reload_tolerance,
        command=command,
    )


def train_classifier(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    mode: FeatureMode,
    classifier: ClassifierName,
    frequency_cache_path: str | Path | None = None,
    clip_cache_path: str | Path | None = None,
    seed: int = 42,
    threshold: float = 0.5,
    max_iter: int = 1000,
    c_value: float = 1.0,
    verify_reload: bool = True,
    reload_tolerance: float = 1e-12,
    command: list[str] | None = None,
) -> TrainResult:
    manifest = Path(manifest_path)
    experiment_dir = Path(output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest)
    validate_manifest_rows(rows, strict=True)
    assembled = assemble_features(
        mode,
        rows,
        frequency_cache_path=frequency_cache_path,
        clip_cache_path=clip_cache_path,
    )

    train_features = assembled.features[assembled.train_mask]
    train_labels = assembled.labels[assembled.train_mask]
    _validate_binary_train_labels(train_labels, classifier)

    if classifier == "logistic_regression":
        model = LogisticRegression(random_state=int(seed), max_iter=int(max_iter), C=float(c_value), solver="lbfgs")
        model.fit(train_features, train_labels)
        prediction = _predict_logistic_regression(model, assembled.features, threshold=threshold)
        classifier_config: dict[str, Any] = {
            "type": CLASSIFIER_NAME,
            "key": classifier,
            "max_iter": int(max_iter),
            "C": float(c_value),
            "probability_supported": True,
            "decision_score_only": False,
            "streamlit_probability_eligible": True,
        }
        calibration = {"enabled": False, "probability_supported": True, "decision_score_only": False}
    else:
        model = LinearSVC(random_state=int(seed), max_iter=int(max_iter), C=float(c_value))
        model.fit(train_features, train_labels)
        prediction = _predict_linear_svm_score_only(model, assembled.features)
        classifier_config = {
            "type": "LinearSVM",
            "key": classifier,
            "max_iter": int(max_iter),
            "C": float(c_value),
            "probability_supported": False,
            "decision_score_only": True,
            "streamlit_probability_eligible": False,
            "limitation": "Linear SVM artifacts expose decision scores only; probabilities are not calibrated or faked.",
        }
        calibration = {
            "enabled": False,
            "method": None,
            "probability_supported": False,
            "decision_score_only": True,
            "streamlit_probability_eligible": False,
            "limitation": "No calibrated probability model was fit; prob_fake is blank in predictions.csv.",
        }

    predictions = _prediction_rows(assembled, prediction)
    metrics = _metrics_by_split(
        assembled.labels,
        prediction.pred_label,
        prediction.prob_fake,
        prediction.score,
        assembled.splits,
        threshold=threshold,
        probability_supported=prediction.probability_supported,
        decision_score_only=prediction.decision_score_only,
    )

    model_path = experiment_dir / "model.joblib"
    scaler_path = experiment_dir / "scaler.joblib"
    config_path = experiment_dir / "config.yaml"
    metrics_path = experiment_dir / "metrics.json"
    predictions_path = experiment_dir / "predictions.csv"

    joblib.dump(model, model_path)
    joblib.dump(assembled.transformers, scaler_path)
    _write_predictions(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    config = _config_snapshot(
        manifest_path=manifest,
        output_dir=experiment_dir,
        mode=mode,
        classifier_config=classifier_config,
        calibration=calibration,
        assembled_metadata=assembled.metadata,
        transformer_keys=sorted(assembled.transformers.keys()),
        model_path=model_path,
        scaler_path=scaler_path,
        threshold=threshold,
        seed=seed,
        command=command or sys.argv,
        frequency_cache_path=frequency_cache_path,
        clip_cache_path=clip_cache_path,
    )
    _write_yaml(config_path, config)

    reload_max_abs_diff = None
    if verify_reload:
        reload_max_abs_diff = verify_reload_equivalence(
            manifest_path=manifest,
            output_dir=experiment_dir,
            mode=mode,
            frequency_cache_path=frequency_cache_path,
            clip_cache_path=clip_cache_path,
            tolerance=reload_tolerance,
        )

    return TrainResult(
        output_dir=experiment_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        reload_max_abs_diff=reload_max_abs_diff,
    )


def verify_reload_equivalence(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    feature_cache_path: str | Path | None = None,
    mode: FeatureMode | None = None,
    frequency_cache_path: str | Path | None = None,
    clip_cache_path: str | Path | None = None,
    tolerance: float = 1e-12,
) -> float:
    experiment_dir = Path(output_dir)
    with (experiment_dir / "config.yaml").open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    resolved_mode = mode or config.get("mode") or "frequency_only"
    resolved_frequency_cache = frequency_cache_path or feature_cache_path or config.get("inputs", {}).get("frequency_cache_path")
    resolved_clip_cache = clip_cache_path or config.get("inputs", {}).get("clip_cache_path")

    rows = read_manifest(manifest_path)
    validate_manifest_rows(rows, strict=True)
    assembled = assemble_features(
        resolved_mode,
        rows,
        frequency_cache_path=resolved_frequency_cache,
        clip_cache_path=resolved_clip_cache,
    )
    model = joblib.load(experiment_dir / "model.joblib")
    classifier_key = str(config.get("classifier", {}).get("key", "logistic_regression"))
    threshold = float(config.get("threshold", 0.5))
    if classifier_key == "linear_svm":
        prediction = _predict_linear_svm_score_only(model, assembled.features)
    else:
        prediction = _predict_logistic_regression(model, assembled.features, threshold=threshold)

    expected = _read_prediction_vectors(experiment_dir / "predictions.csv")
    diffs = []
    if prediction.prob_fake is not None and expected["prob_fake"].size:
        diffs.append(float(np.max(np.abs(prediction.prob_fake - expected["prob_fake"]))))
    if prediction.score.size:
        diffs.append(float(np.max(np.abs(prediction.score - expected["score"]))))
    max_abs_diff = max(diffs) if diffs else 0.0
    if max_abs_diff > float(tolerance):
        raise ValueError(f"reload predictions differ by {max_abs_diff:.6g}, tolerance {tolerance:.6g}")
    return max_abs_diff


@dataclass(frozen=True)
class _PredictionResult:
    pred_label: np.ndarray
    prob_fake: np.ndarray | None
    score: np.ndarray
    probability_supported: bool
    decision_score_only: bool


def _validate_binary_train_labels(labels: np.ndarray, classifier: str) -> None:
    if set(np.unique(labels).astype(int).tolist()) != {0, 1}:
        raise ValueError(f"train split must contain both labels 0 and 1 for {classifier}")


def _predict_logistic_regression(model: LogisticRegression, features: np.ndarray, *, threshold: float) -> _PredictionResult:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    matching = np.flatnonzero(classes == 1)
    if matching.size != 1:
        raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
    fake_index = int(matching[0])
    prob_fake = probabilities[:, fake_index].astype(np.float64, copy=False)
    pred_label = (prob_fake >= float(threshold)).astype(np.int64)
    score = _positive_class_score(model, features, classes)
    return _PredictionResult(pred_label, prob_fake, score, True, False)


def _predict_linear_svm_score_only(model: LinearSVC, features: np.ndarray) -> _PredictionResult:
    classes = np.asarray(model.classes_, dtype=np.int64)
    score = _positive_class_score(model, features, classes)
    pred_label = (score >= 0.0).astype(np.int64)
    return _PredictionResult(pred_label, None, score, False, True)


def _positive_class_score(model: Any, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
    decision = model.decision_function(features)
    if np.asarray(decision).ndim == 1:
        if classes.shape[0] != 2:
            raise ValueError(f"binary decision score expected 2 classes, got {classes.tolist()}")
        values = np.asarray(decision, dtype=np.float64)
        return values if int(classes[1]) == 1 else -values
    decision_matrix = np.asarray(decision, dtype=np.float64)
    matching = np.flatnonzero(classes == 1)
    if matching.size != 1:
        raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
    return decision_matrix[:, int(matching[0])].astype(np.float64, copy=False)


def _prediction_rows(assembled: Any, prediction: _PredictionResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, sample_id in enumerate(assembled.sample_ids):
        prob_fake = "" if prediction.prob_fake is None else f"{float(prediction.prob_fake[index]):.17g}"
        rows.append(
            {
                "path": str(assembled.paths[index]),
                "sample_id": str(sample_id),
                "label": str(int(assembled.labels[index])),
                "pred_label": str(int(prediction.pred_label[index])),
                "prob_fake": prob_fake,
                "score": f"{float(prediction.score[index]):.17g}",
                "split": str(assembled.splits[index]),
            }
        )
    return rows


def _metrics_by_split(
    labels: np.ndarray,
    pred_label: np.ndarray,
    prob_fake: np.ndarray | None,
    score: np.ndarray,
    splits: np.ndarray,
    *,
    threshold: float,
    probability_supported: bool,
    decision_score_only: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "probability_supported": bool(probability_supported),
        "decision_score_only": bool(decision_score_only),
        "splits": {},
    }
    ranking_values = prob_fake if prob_fake is not None else score
    for split in ["train", "val", "test"]:
        mask = splits == split
        metrics["splits"][split] = _split_metrics(
            labels[mask], pred_label[mask], ranking_values[mask], probability_supported=probability_supported
        )
    metrics["overall"] = _split_metrics(labels, pred_label, ranking_values, probability_supported=probability_supported)
    return metrics


def _split_metrics(
    labels: np.ndarray, pred_label: np.ndarray, ranking_values: np.ndarray, *, probability_supported: bool
) -> dict[str, Any]:
    row_count = int(labels.shape[0])
    if row_count == 0:
        return {"sample_count": 0, "available": False, "reason": "split has no rows"}
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_label, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    result: dict[str, Any] = {
        "sample_count": row_count,
        "available": True,
        "label_counts": {str(label): int(np.sum(labels == label)) for label in [0, 1]},
        "accuracy": float(accuracy_score(labels, pred_label)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "probability_supported": bool(probability_supported),
    }
    if set(np.unique(labels).astype(int).tolist()) == {0, 1}:
        result["roc_auc"] = float(roc_auc_score(labels, ranking_values))
        result["average_precision"] = float(average_precision_score(labels, ranking_values))
        result["ranking_metric_input"] = "prob_fake" if probability_supported else "decision_score"
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
        result["ranking_metrics_available"] = False
        result["ranking_metrics_reason"] = "split does not contain both labels 0 and 1"
    return _finite_json(result)


def _config_snapshot(
    *,
    manifest_path: Path,
    output_dir: Path,
    mode: FeatureMode,
    classifier_config: dict[str, Any],
    calibration: dict[str, Any],
    assembled_metadata: dict[str, Any],
    transformer_keys: list[str],
    model_path: Path,
    scaler_path: Path,
    threshold: float,
    seed: int,
    command: list[str],
    frequency_cache_path: str | Path | None,
    clip_cache_path: str | Path | None,
) -> dict[str, Any]:
    cache_hashes: dict[str, Any] = {}
    for branch, metadata in assembled_metadata.get("branches", {}).items():
        cache_hashes[str(branch)] = {
            "manifest_hash": str(metadata.get("manifest_hash", "")),
            "feature_config_hash": str(metadata.get("feature_config_hash", "")),
        }
    return {
        "mode": mode,
        "classifier": classifier_config,
        "calibration": calibration,
        "probability_supported": bool(classifier_config["probability_supported"]),
        "decision_score_only": bool(classifier_config["decision_score_only"]),
        "streamlit_probability_eligible": bool(classifier_config["streamlit_probability_eligible"]),
        "threshold": float(threshold),
        "label_mapping": dict(CLASS_TO_LABEL),
        "feature": assembled_metadata,
        "transformers": {"path": scaler_path.as_posix(), "keys": transformer_keys},
        "artifacts": {
            "output_dir": output_dir.as_posix(),
            "model_path": model_path.as_posix(),
            "scaler_path": scaler_path.as_posix(),
        },
        "hashes": {"caches": cache_hashes},
        "inputs": {
            "manifest_path": manifest_path.as_posix(),
            "frequency_cache_path": None if frequency_cache_path is None else Path(frequency_cache_path).as_posix(),
            "clip_cache_path": None if clip_cache_path is None else Path(clip_cache_path).as_posix(),
        },
        "training_command": list(command),
        "seed": int(seed),
        "package_metadata": _package_metadata(),
        "device_metadata": {"device": "cpu"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _package_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "joblib": joblib.__version__,
        "pyyaml": yaml.__version__,
    }


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PREDICTION_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(_finite_json(payload), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(_finite_json(payload), file_obj, sort_keys=False)


def _read_prediction_vectors(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    columns = list(rows[0].keys()) if rows else PREDICTION_COLUMNS
    if columns != PREDICTION_COLUMNS:
        raise ValueError(f"predictions.csv columns must be {PREDICTION_COLUMNS}, got {columns}")
    prob_values = [float(row["prob_fake"]) for row in rows if row["prob_fake"] != ""]
    return {
        "prob_fake": np.asarray(prob_values, dtype=np.float64),
        "score": np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
    }


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite metric value: {value}")
        return value
    return value
