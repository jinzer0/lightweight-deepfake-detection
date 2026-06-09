from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportArgumentType=false

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

PREDICTION_COLUMNS = ["path", "sample_id", "label", "pred_label", "prob_fake", "score", "split"]
REQUIRED_ARTIFACT_FILES = [
    "model.joblib",
    "scaler.joblib",
    "config.yaml",
    "metrics.json",
    "predictions.csv",
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
]


class ArtifactValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def evaluate_experiment(experiment_dir: str | Path, *, split: str | None = None, validate: bool = False) -> dict[str, Any]:
    directory = Path(experiment_dir)
    config = _read_yaml(directory / "config.yaml")
    rows = _read_predictions(directory / "predictions.csv")
    selected_rows = _filter_rows(rows, split)
    vectors = _prediction_vectors(selected_rows, config)
    metrics = compute_metrics(vectors, config, split=split)

    _write_json(directory / "metrics.json", metrics)
    _write_plots(directory, vectors, metrics)
    if validate:
        validate_experiment_artifacts(directory)
    return metrics


def validate_experiment_artifacts(experiment_dir: str | Path) -> list[str]:
    directory = Path(experiment_dir)
    errors: list[str] = []
    for file_name in REQUIRED_ARTIFACT_FILES:
        path = directory / file_name
        if not path.exists():
            errors.append(f"missing required artifact: {file_name}")
        elif file_name.endswith(".png") and path.stat().st_size == 0:
            errors.append(f"plot artifact is empty: {file_name}")

    config = _safe_read_yaml(directory / "config.yaml", errors)
    rows = _safe_read_predictions(directory / "predictions.csv", errors)
    metrics = _safe_read_json(directory / "metrics.json", errors)
    if rows is not None:
        _validate_prediction_rows(rows, config or {}, errors)
        _validate_row_count(rows, config or {}, errors)
    if metrics is not None:
        _validate_metrics(metrics, errors)

    if errors:
        raise ArtifactValidationError(errors)
    return []


def compute_metrics(vectors: dict[str, np.ndarray], config: dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    labels = vectors["label"]
    pred_labels = vectors["pred_label"]
    ranking_values = vectors["ranking"]
    probability_supported = bool(config.get("probability_supported", False))
    decision_score_only = bool(config.get("decision_score_only", False))
    ranking_metric_input = "prob_fake" if probability_supported else "decision_score"
    metrics: dict[str, Any] = {
        "sample_count": int(labels.shape[0]),
        "split": split or "all",
        "threshold": float(config.get("threshold", 0.5)),
        "probability_supported": probability_supported,
        "decision_score_only": decision_score_only,
        "ranking_metric_input": ranking_metric_input,
        "overall": _single_metrics(labels, pred_labels, ranking_values, ranking_metric_input),
        "splits": {},
    }
    splits = vectors["split"]
    for split_name in sorted({str(value) for value in splits.tolist()}):
        mask = splits == split_name
        metrics["splits"][split_name] = _single_metrics(
            labels[mask], pred_labels[mask], ranking_values[mask], ranking_metric_input
        )
    return _finite_json(metrics)


def _single_metrics(
    labels: np.ndarray, pred_labels: np.ndarray, ranking_values: np.ndarray, ranking_metric_input: str
) -> dict[str, Any]:
    sample_count = int(labels.shape[0])
    if sample_count == 0:
        return {"sample_count": 0, "available": False, "reason": "split has no rows"}
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_labels, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    matrix = confusion_matrix(labels, pred_labels, labels=[0, 1]).astype(int).tolist()
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "available": True,
        "label_counts": {str(label): int(np.sum(labels == label)) for label in [0, 1]},
        "accuracy": float(accuracy_score(labels, pred_labels)),
        "precision": float(precision),
        "precision_fake": float(precision),
        "recall": float(recall),
        "recall_fake": float(recall),
        "f1": float(f1),
        "f1_fake": float(f1),
        "confusion_matrix": matrix,
        "ranking_metric_input": ranking_metric_input,
    }
    if set(np.unique(labels).astype(int).tolist()) == {0, 1}:
        result["roc_auc"] = float(roc_auc_score(labels, ranking_values))
        result["average_precision"] = float(average_precision_score(labels, ranking_values))
        result["pr_auc"] = result["average_precision"]
        result["ranking_metrics_available"] = True
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
        result["pr_auc"] = None
        result["ranking_metrics_available"] = False
        result["ranking_metrics_reason"] = "split does not contain both labels 0 and 1"
    return result


def _write_plots(directory: Path, vectors: dict[str, np.ndarray], metrics: dict[str, Any]) -> None:
    labels = vectors["label"]
    pred_labels = vectors["pred_label"]
    ranking_values = vectors["ranking"]
    matrix = confusion_matrix(labels, pred_labels, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["real", "fake"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    ranking_available = bool(metrics["overall"].get("ranking_metrics_available"))
    _write_ranking_plot(directory / "roc_curve.png", labels, ranking_values, ranking_available, kind="roc")
    _write_ranking_plot(directory / "pr_curve.png", labels, ranking_values, ranking_available, kind="pr")


def _write_ranking_plot(path: Path, labels: np.ndarray, ranking_values: np.ndarray, ranking_available: bool, *, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    if ranking_available:
        if kind == "roc":
            RocCurveDisplay.from_predictions(labels, ranking_values, ax=ax)
            ax.set_title("ROC Curve")
        else:
            PrecisionRecallDisplay.from_predictions(labels, ranking_values, ax=ax)
            ax.set_title("Precision-Recall Curve")
    else:
        ax.text(0.5, 0.5, "Ranking metric unavailable\nrequires both labels", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"predictions.csv columns must be exactly {PREDICTION_COLUMNS}, got {reader.fieldnames}")
        return list(reader)


def _prediction_vectors(rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("predictions.csv has no rows")
    probability_supported = bool(config.get("probability_supported", False))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    pred_labels = np.asarray([int(row["pred_label"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if probability_supported:
        ranking = np.asarray([float(row["prob_fake"]) for row in rows], dtype=np.float64)
    else:
        ranking = scores
    splits = np.asarray([str(row["split"]) for row in rows], dtype=object)
    return {"label": labels, "pred_label": pred_labels, "score": scores, "ranking": ranking, "split": splits}


def _filter_rows(rows: list[dict[str, str]], split: str | None) -> list[dict[str, str]]:
    if split is None:
        return rows
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"predictions.csv has no rows for split {split!r}")
    return selected


def _validate_prediction_rows(rows: list[dict[str, str]], config: dict[str, Any], errors: list[str]) -> None:
    if not rows:
        errors.append("predictions.csv has no rows")
        return
    probability_supported = bool(config.get("probability_supported", False))
    decision_score_only = bool(config.get("decision_score_only", False))
    seen_sample_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if set(row.keys()) != set(PREDICTION_COLUMNS):
            errors.append(f"predictions.csv row {index} has invalid columns")
        if row.get("sample_id", "") in seen_sample_ids:
            errors.append(f"predictions.csv row {index} duplicates sample_id {row.get('sample_id')!r}")
        seen_sample_ids.add(row.get("sample_id", ""))
        for column in ["label", "pred_label"]:
            if row.get(column) not in {"0", "1"}:
                errors.append(f"predictions.csv row {index} {column} must be 0 or 1")
        score = _parse_float(row.get("score", ""))
        if score is None:
            errors.append(f"predictions.csv row {index} score must be finite")
        prob_fake = row.get("prob_fake", "")
        if probability_supported:
            probability = _parse_float(prob_fake)
            if probability is None or probability < 0.0 or probability > 1.0:
                errors.append(f"predictions.csv row {index} prob_fake must be finite in [0, 1]")
        elif prob_fake != "" and decision_score_only:
            errors.append(f"predictions.csv row {index} prob_fake must be blank for decision_score_only artifacts")


def _validate_row_count(rows: list[dict[str, str]], config: dict[str, Any], errors: list[str]) -> None:
    manifest_path_value = config.get("inputs", {}).get("manifest_path") if isinstance(config.get("inputs"), dict) else None
    if not manifest_path_value:
        return
    manifest_path = Path(str(manifest_path_value))
    if not manifest_path.exists():
        return
    with manifest_path.open("r", newline="", encoding="utf-8") as file_obj:
        expected_count = sum(1 for _ in csv.DictReader(file_obj))
    if len(rows) != expected_count:
        errors.append(f"predictions.csv row count {len(rows)} does not match manifest row count {expected_count}")


def _validate_metrics(metrics: Any, errors: list[str]) -> None:
    if not isinstance(metrics, dict):
        errors.append("metrics.json must contain an object")
        return
    required = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "confusion_matrix"]
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        errors.append("metrics.json missing overall metrics object")
        return
    for key in required:
        if key not in overall:
            errors.append(f"metrics.json overall missing {key}")
    _check_finite_metrics(metrics, "metrics", errors)


def _check_finite_metrics(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _check_finite_metrics(nested, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _check_finite_metrics(nested, f"{path}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} must be finite")


def _safe_read_predictions(path: Path, errors: list[str]) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        return _read_predictions(path)
    except (OSError, ValueError) as error:
        errors.append(str(error))
        return None


def _safe_read_yaml(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_yaml(path)
    except (OSError, yaml.YAMLError, TypeError) as error:
        errors.append(f"config.yaml is invalid: {error}")
        return None


def _safe_read_json(path: Path, errors: list[str]) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"metrics.json is invalid: {error}")
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    if not isinstance(data, dict):
        raise TypeError("config.yaml must contain an object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_finite_json(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
