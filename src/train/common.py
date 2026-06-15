from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import csv
import json
import math
import warnings
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.features.cache_features import NpyFeatureCacheError, cache_paths, load_feature_cache
from src.models.checkpoint import save_checkpoint
from src.models.mlp_classifier import MLPClassifier
from src.utils.config import resolve_device
from src.utils.seed import set_seed

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is a progress nicety
    tqdm = None


T = TypeVar("T")


LOG_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_roc_auc",
]


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    train_log_path: Path
    val_metrics_path: Path
    best_epoch: int
    best_val_loss: float
    best_val_roc_auc: float | None


@dataclass(frozen=True)
class TrainerSettings:
    feature_type: str
    artifact_stem: str
    optional_cache: bool = False


def train_feature_mlp(config: dict[str, object], settings: TrainerSettings) -> TrainingArtifacts:
    seed = _project_seed(config)
    set_seed(seed)
    device = torch.device(resolve_device(config))

    train_features, train_labels = _load_split(config, feature_type=settings.feature_type, split="train", optional_cache=settings.optional_cache)
    val_features, val_labels = _load_split(config, feature_type=settings.feature_type, split="val", optional_cache=settings.optional_cache)
    checkpoint_config = dict(config)
    if settings.feature_type == "frequency":
        scaler, scaler_path = _fit_frequency_scaler(config, train_features)
        train_features = cast(np.ndarray, scaler.transform(train_features)).astype(np.float32, copy=False)
        val_features = cast(np.ndarray, scaler.transform(val_features)).astype(np.float32, copy=False)
        checkpoint_config = _config_with_frequency_scaler(config, scaler_path)
    _validate_training_labels(train_labels, split="train")

    train_loader = _make_loader(train_features, train_labels, batch_size=_batch_size(config), shuffle=True, seed=seed)
    val_loader = _make_loader(val_features, val_labels, batch_size=_batch_size(config), shuffle=False, seed=seed)

    input_dim = int(train_features.shape[1])
    hidden_dim = _hidden_dim(config)
    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=_dropout(config)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate(config), weight_decay=_weight_decay(config))

    checkpoint_path = _checkpoint_path(config, settings.artifact_stem)
    train_log_path = _report_path(config, f"{settings.artifact_stem}_train_log.csv")
    val_metrics_path = _report_path(config, f"{settings.artifact_stem}_val_metrics.json")

    rows: list[dict[str, object]] = []
    best_score = (-math.inf, -math.inf)
    best_epoch = 0
    best_val_loss = math.inf
    best_val_roc_auc: float | None = None
    best_metrics: dict[str, object] = {}
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    epoch_iter = _progress_iter(range(1, _epochs(config) + 1), desc=f"Training {settings.artifact_stem}", unit="epoch")
    for epoch in epoch_iter:
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = _evaluate(model, val_loader, criterion, device, threshold=_threshold(config))
        score = _selection_score(val_loss, metrics["roc_auc"])
        rows.append(_log_row(epoch, train_loss, val_loss, metrics))
        _set_progress_postfix(epoch_iter, train_loss=train_loss, val_loss=val_loss, val_accuracy=metrics["accuracy"], val_roc_auc=metrics["roc_auc"])

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_roc_auc = _optional_float(metrics["roc_auc"])
            best_metrics = dict(metrics)
            best_metrics["loss"] = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= _patience(config):
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")

    _write_train_log(train_log_path, rows)
    _write_val_metrics(
        val_metrics_path,
        metrics=best_metrics,
        feature_type=settings.feature_type,
        model_name="MLPClassifier",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        threshold=_threshold(config),
        best_epoch=best_epoch,
    )
    save_checkpoint(
        checkpoint_path,
        model_state_dict=best_state,
        model_name="MLPClassifier",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        threshold=_threshold(config),
        feature_type=settings.feature_type,
        config_snapshot=checkpoint_config,
    )

    print(f"best_epoch={best_epoch}")
    print(f"best_val_loss={best_val_loss:.6f}")
    print(f"best_val_roc_auc={best_val_roc_auc if best_val_roc_auc is not None else 'null'}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved train log: {train_log_path}")
    print(f"saved val metrics: {val_metrics_path}")

    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        train_log_path=train_log_path,
        val_metrics_path=val_metrics_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_roc_auc=best_val_roc_auc,
    )


def _progress_iter(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None, leave: bool = True) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def _set_progress_postfix(progress: Iterable[object], **metrics: object) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if set_postfix is None:
        return
    formatted = {key: _format_progress_value(value) for key, value in metrics.items()}
    set_postfix(formatted)


def _format_progress_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else str(value)
    if value is None:
        return "null"
    return value


def _load_split(config: dict[str, object], *, feature_type: str, split: str, optional_cache: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        features, labels, _meta = load_feature_cache(config, feature_type=feature_type, split=split)
    except NpyFeatureCacheError as exc:
        paths = cache_paths(config, feature_type=feature_type, split=split)
        prefix = "Optional CLIP cache is missing" if optional_cache and feature_type == "clip" else "Feature cache is missing"
        raise NpyFeatureCacheError(
            f"{prefix} for feature_type={feature_type!r} split={split!r}; expected features={paths.features} labels={paths.labels}. {exc}"
        ) from exc
    if features.shape[0] == 0:
        raise NpyFeatureCacheError(f"{feature_type} {split} cache contains no rows")
    return features.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


def _make_loader(features: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader[tuple[torch.Tensor, ...]]:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    label_tensor = torch.from_numpy(labels.astype(np.float32, copy=False))
    dataset = TensorDataset(feature_tensor, label_tensor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = cast(torch.Tensor, model(features))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_rows = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_rows
        total_rows += batch_rows
    return total_loss / max(total_rows, 1)


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    *,
    threshold: float,
) -> tuple[float, dict[str, float | None]]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = cast(torch.Tensor, model(features))
            loss = criterion(logits, labels)
            batch_rows = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
            logits_parts.append(logits.detach().cpu().numpy())
            label_parts.append(labels.detach().cpu().numpy())
    logits_array = np.concatenate(logits_parts).astype(np.float64, copy=False)
    labels_array = np.concatenate(label_parts).astype(np.int64, copy=False)
    probabilities = 1.0 / (1.0 + np.exp(-logits_array))
    return total_loss / max(total_rows, 1), compute_binary_metrics(labels_array, probabilities, threshold=threshold)


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, *, threshold: float) -> dict[str, float | None]:
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    labels = labels.astype(np.int64, copy=False)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    total = int(labels.size)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": _roc_auc(labels, probabilities),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        warnings.warn("validation ROC-AUC is undefined because one class is present; writing null", RuntimeWarning, stacklevel=2)
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _selection_score(val_loss: float, roc_auc: float | None) -> tuple[float, float]:
    if roc_auc is not None and math.isfinite(float(roc_auc)):
        return (float(roc_auc), -float(val_loss))
    return (-math.inf, -float(val_loss))


def _log_row(epoch: int, train_loss: float, val_loss: float, metrics: dict[str, float | None]) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": metrics["accuracy"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_roc_auc": metrics["roc_auc"],
    }


def _write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_val_metrics(
    path: Path,
    *,
    metrics: dict[str, object],
    feature_type: str,
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    threshold: float,
    best_epoch: int,
) -> None:
    payload = {
        "feature_type": feature_type,
        "model_name": model_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _validate_training_labels(labels: np.ndarray, *, split: str) -> None:
    unique = set(labels.astype(int).tolist())
    if unique != {0, 1}:
        raise ValueError(f"{split} split must contain both labels 0 and 1 for BCEWithLogitsLoss training; got {sorted(unique)}")


def _paths(config: dict[str, object]) -> dict[str, object]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    return paths


def _checkpoint_path(config: dict[str, object], artifact_stem: str) -> Path:
    paths = _paths(config)
    return Path(str(paths.get("checkpoint_dir", "artifacts/checkpoints"))) / f"{artifact_stem}.pt"


def _frequency_scaler_path(config: dict[str, object]) -> Path:
    paths = _paths(config)
    explicit = paths.get("frequency_scaler_path")
    if explicit:
        return Path(str(explicit))
    scaler_dir = Path(str(paths.get("scaler_dir", "artifacts/scalers")))
    return scaler_dir / "frequency_scaler.pkl"


def _fit_frequency_scaler(config: dict[str, object], train_features: np.ndarray) -> tuple[StandardScaler, Path]:
    scaler = StandardScaler()
    scaler.fit(train_features.astype(np.float32, copy=False))
    scaler_path = _frequency_scaler_path(config)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaler, scaler_path


def _config_with_frequency_scaler(config: dict[str, object], scaler_path: Path) -> dict[str, object]:
    snapshot = dict(config)
    raw_paths = snapshot.get("paths")
    paths = dict(cast(dict[str, object], raw_paths)) if isinstance(raw_paths, dict) else {}
    paths["frequency_scaler_path"] = scaler_path.as_posix()
    snapshot["paths"] = paths
    raw_frequency = snapshot.get("frequency")
    frequency = dict(cast(dict[str, object], raw_frequency)) if isinstance(raw_frequency, dict) else {}
    frequency["scaler"] = "standard"
    snapshot["frequency"] = frequency
    return snapshot


def _report_path(config: dict[str, object], filename: str) -> Path:
    paths = _paths(config)
    return Path(str(paths.get("report_dir", "artifacts/reports"))) / filename


def _project_seed(config: dict[str, object]) -> int:
    project = config.get("project")
    if isinstance(project, dict):
        return int(project.get("seed", 42))
    return 42


def _batch_size(config: dict[str, object]) -> int:
    data = config.get("data")
    if isinstance(data, dict):
        return int(data.get("batch_size", 32))
    return 32


def _hidden_dim(config: dict[str, object]) -> int:
    classifier = config.get("classifier")
    if isinstance(classifier, dict):
        return int(classifier.get("hidden_dim", 256))
    return 256


def _dropout(config: dict[str, object]) -> float:
    classifier = config.get("classifier")
    if isinstance(classifier, dict):
        return float(classifier.get("dropout", 0.2))
    return 0.2


def _epochs(config: dict[str, object]) -> int:
    train = config.get("train")
    if isinstance(train, dict):
        return int(train.get("epochs", 20))
    return 20


def _learning_rate(config: dict[str, object]) -> float:
    train = config.get("train")
    if isinstance(train, dict):
        return float(train.get("learning_rate", 1e-4))
    return 1e-4


def _weight_decay(config: dict[str, object]) -> float:
    train = config.get("train")
    if isinstance(train, dict):
        return float(train.get("weight_decay", 1e-4))
    return 1e-4


def _patience(config: dict[str, object]) -> int:
    train = config.get("train")
    if isinstance(train, dict):
        return int(train.get("early_stopping_patience", 5))
    return 5


def _threshold(config: dict[str, object]) -> float:
    evaluation = config.get("eval")
    if isinstance(evaluation, dict):
        return float(evaluation.get("threshold", 0.5))
    return 0.5


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected numeric metric or None, got {type(value).__name__}")
