from __future__ import annotations

# pyright: reportAny=false, reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportExplicitAny=false, reportUnusedCallResult=false

import argparse
import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.eval.metrics import compute_binary_metrics
from src.features.cache_features import NpyFeatureCacheError, load_feature_cache
from src.models.checkpoint import CheckpointError, load_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.models.mlp_classifier import MLPClassifier
from src.train.train_fusion import align_feature_tables
from src.utils.config import load_config, resolve_device


MODEL_SPECS = {
    "frequency_only": {"checkpoint": "frequency_only.pt", "feature_type": "frequency", "class": "MLPClassifier"},
    "clip_only": {"checkpoint": "clip_only.pt", "feature_type": "clip", "class": "MLPClassifier"},
    "fusion": {"checkpoint": "fusion.pt", "feature_type": "fusion", "class": "FusionClassifier"},
}
PREDICTION_COLUMNS = ["image_id", "filepath", "label", "pred_prob", "pred_label", "class_name", "dataset", "generator", "split", "model_name"]
PER_GENERATOR_COLUMNS = ["model_name", "generator", "count", "accuracy", "precision", "recall", "f1", "roc_auc"]
COMPARISON_COLUMNS = ["model_name", "split", "sample_count", "accuracy", "precision", "recall", "f1", "roc_auc", "metrics_path"]


@dataclass(frozen=True)
class EvaluationResult:
    metrics_path: Path
    predictions_path: Path
    per_generator_path: Path
    comparison_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class EvaluationTable:
    features: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame
    clip_dim: int | None = None
    frequency_dim: int | None = None


def evaluate_model(config: Mapping[str, object], *, model_name: str, split: str) -> EvaluationResult:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"model must be one of {sorted(MODEL_SPECS)}, got {model_name!r}")
    spec = MODEL_SPECS[model_name]
    feature_type = str(spec["feature_type"])
    table = _load_evaluation_table(config, model_name=model_name, split=split)
    checkpoint_path = _checkpoint_dir(config) / str(spec["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, expected_feature_type=feature_type)
    threshold = _float_checkpoint_value(checkpoint["threshold"], "threshold")
    model = _build_model(checkpoint, table=table, model_name=model_name)
    device = torch.device(resolve_device(dict(config)))
    model.to(device)
    probabilities = _predict_probabilities(model, table.features, device=device)
    pred_labels = (probabilities >= threshold).astype(np.int64)

    overall = compute_binary_metrics(table.labels, probabilities, threshold=threshold)
    per_generator = _per_generator_metrics(table.meta, table.labels, probabilities, threshold=threshold, model_name=model_name)
    metrics = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path.as_posix(),
        "feature_type": feature_type,
        "split": split,
        "threshold": threshold,
        "sample_count": int(table.labels.shape[0]),
        "metrics": overall,
        "per_generator": per_generator,
    }

    report_dir = _report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / f"{model_name}_{split}_metrics.json"
    predictions_path = report_dir / f"{model_name}_{split}_predictions.csv"
    per_generator_path = report_dir / f"{model_name}_{split}_per_generator_metrics.csv"
    comparison_path = report_dir / "model_comparison.csv"
    _write_json(metrics_path, metrics)
    _write_predictions(predictions_path, table.meta, table.labels, probabilities, pred_labels, model_name=model_name)
    _write_per_generator(per_generator_path, per_generator)
    write_model_comparison(report_dir)
    return EvaluationResult(metrics_path, predictions_path, per_generator_path, comparison_path, metrics)


def write_model_comparison(report_dir: str | Path) -> Path:
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*_metrics.json")):
        payload = _read_json(path)
        if not _is_evaluator_metrics(payload):
            continue
        metrics = cast(dict[str, object], payload["metrics"])
        rows.append(
            {
                "model_name": str(payload["model_name"]),
                "split": str(payload["split"]),
                "sample_count": _int_report_value(payload.get("sample_count", metrics.get("sample_count", 0))),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "metrics_path": path.as_posix(),
            }
        )
    comparison_path = directory / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return comparison_path


def _load_evaluation_table(config: Mapping[str, object], *, model_name: str, split: str) -> EvaluationTable:
    if model_name == "fusion":
        table = align_feature_tables(config, split=split)
        return EvaluationTable(
            features=table.features.astype(np.float32, copy=False),
            labels=table.labels.astype(np.int64, copy=False),
            meta=table.meta.reset_index(drop=True),
            clip_dim=int(table.clip_features.shape[1]),
            frequency_dim=int(table.frequency_features.shape[1]),
        )
    feature_type = str(MODEL_SPECS[model_name]["feature_type"])
    features, labels, meta = load_feature_cache(config, feature_type=feature_type, split=split)
    return EvaluationTable(features=features.astype(np.float32, copy=False), labels=labels.astype(np.int64, copy=False), meta=meta.reset_index(drop=True))


def _build_model(checkpoint: Mapping[str, object], *, table: EvaluationTable, model_name: str) -> nn.Module:
    input_dim = _int_checkpoint_value(checkpoint["input_dim"], "input_dim")
    hidden_dim = _int_checkpoint_value(checkpoint["hidden_dim"], "hidden_dim")
    if input_dim != int(table.features.shape[1]):
        raise CheckpointError(f"checkpoint input_dim {input_dim} does not match evaluated feature dimension {table.features.shape[1]}")
    checkpoint_model_name = str(checkpoint["model_name"])
    if model_name == "fusion":
        if checkpoint_model_name != "FusionClassifier":
            raise CheckpointError(f"fusion checkpoint model_name must be FusionClassifier, got {checkpoint_model_name!r}")
        if table.clip_dim is None or table.frequency_dim is None:
            raise CheckpointError("fusion evaluation requires aligned CLIP and frequency dimensions")
        model: nn.Module = FusionClassifier(clip_dim=table.clip_dim, freq_dim=table.frequency_dim, hidden_dim=hidden_dim, dropout=0.0)
    else:
        if checkpoint_model_name != "MLPClassifier":
            raise CheckpointError(f"{model_name} checkpoint model_name must be MLPClassifier, got {checkpoint_model_name!r}")
        model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state = cast(Mapping[str, torch.Tensor], checkpoint["model_state_dict"])
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_probabilities(model: nn.Module, features: np.ndarray, *, device: torch.device) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"evaluation features must be non-empty 2D array, got shape {features.shape}")
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
    if probabilities.ndim != 1 or probabilities.shape[0] != features.shape[0]:
        raise ValueError(f"model must return one logit per row, got probabilities shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("model produced invalid fake probabilities")
    return probabilities


def _per_generator_metrics(
    meta: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray, *, threshold: float, model_name: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    generators = meta["generator"].astype(str).fillna("unknown")
    for generator in sorted(generators.unique().tolist()):
        mask = generators.to_numpy() == generator
        metrics = compute_binary_metrics(labels[mask], probabilities[mask], threshold=threshold)
        rows.append(
            {
                "model_name": model_name,
                "generator": generator,
                "count": int(np.sum(mask)),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    return rows


def _write_predictions(
    path: Path, meta: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray, pred_labels: np.ndarray, *, model_name: str
) -> None:
    if len(meta) != int(labels.shape[0]) or int(probabilities.shape[0]) != int(labels.shape[0]):
        raise ValueError("prediction row count must match metadata and labels")
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for index, row in meta.reset_index(drop=True).iterrows():
            writer.writerow(
                {
                    "image_id": str(row["image_id"]),
                    "filepath": str(row["filepath"]),
                    "label": int(labels[index]),
                    "pred_prob": f"{float(probabilities[index]):.17g}",
                    "pred_label": int(pred_labels[index]),
                    "class_name": str(row["class_name"]),
                    "dataset": str(row["dataset"]),
                    "generator": str(row["generator"]),
                    "split": str(row["split"]),
                    "model_name": model_name,
                }
            )


def _write_per_generator(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PER_GENERATOR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True, allow_nan=False)
        file_obj.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, object], payload)


def _is_evaluator_metrics(payload: Mapping[str, object]) -> bool:
    return isinstance(payload.get("metrics"), dict) and isinstance(payload.get("model_name"), str) and isinstance(payload.get("split"), str)


def _int_checkpoint_value(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be an int")
    return value


def _float_checkpoint_value(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be numeric")
    return float(value)


def _int_report_value(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _paths(config: Mapping[str, object]) -> Mapping[str, object]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("config.paths must be a mapping")
    return paths


def _checkpoint_dir(config: Mapping[str, object]) -> Path:
    return Path(str(_paths(config).get("checkpoint_dir", "artifacts/checkpoints")))


def _report_dir(config: Mapping[str, object]) -> Path:
    return Path(str(_paths(config).get("report_dir", "artifacts/reports")))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cached-feature PyTorch classifiers and write report artifacts.")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS), help="Model artifact to evaluate")
    parser.add_argument("--split", default="test", help="Cached split to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    try:
        result = evaluate_model(config, model_name=str(args.model), split=str(args.split))
    except (CheckpointError, NpyFeatureCacheError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Evaluation failed clearly: {exc}") from None
    metrics = result.metrics["metrics"]
    print(f"model_name={args.model}")
    print(f"split={args.split}")
    print(f"sample_count={metrics['sample_count']}")
    print(f"accuracy={float(metrics['accuracy']):.6f}")
    print(f"roc_auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'null'}")
    print(f"saved metrics: {result.metrics_path}")
    print(f"saved predictions: {result.predictions_path}")
    print(f"saved per-generator metrics: {result.per_generator_path}")
    print(f"saved model comparison: {result.comparison_path}")


if __name__ == "__main__":
    main()
