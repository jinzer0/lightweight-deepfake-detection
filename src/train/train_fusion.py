from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportPrivateUsage=false, reportUnusedCallResult=false

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.features.cache_features import NpyFeatureCacheError, cache_paths, load_feature_cache
from src.models.checkpoint import save_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.train.common import (
    TrainingArtifacts,
    _batch_size,
    _checkpoint_path,
    _dropout,
    _epochs,
    _evaluate,
    _hidden_dim,
    _learning_rate,
    _log_row,
    _make_loader,
    _optional_float,
    _patience,
    _project_seed,
    _report_path,
    _selection_score,
    _threshold,
    _train_one_epoch,
    _validate_training_labels,
    _weight_decay,
    _write_train_log,
    _write_val_metrics,
)
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed


@dataclass(frozen=True)
class FusionFeatureTable:
    features: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame
    clip_features: np.ndarray
    frequency_features: np.ndarray


def align_feature_tables(
    config: Mapping[str, object] | None = None,
    *,
    split: str,
    clip_features: np.ndarray | None = None,
    clip_labels: np.ndarray | None = None,
    clip_meta: pd.DataFrame | None = None,
    frequency_features: np.ndarray | None = None,
    frequency_labels: np.ndarray | None = None,
    frequency_meta: pd.DataFrame | None = None,
) -> FusionFeatureTable:
    if config is not None:
        clip_features, clip_labels, clip_meta = _load_required_branch(config, feature_type="clip", split=split)
        frequency_features, frequency_labels, frequency_meta = _load_required_branch(config, feature_type="frequency", split=split)

    if clip_features is None or clip_labels is None or clip_meta is None:
        raise TypeError("align_feature_tables requires either config or all CLIP feature/label/meta inputs")
    if frequency_features is None or frequency_labels is None or frequency_meta is None:
        raise TypeError("align_feature_tables requires either config or all frequency feature/label/meta inputs")

    clip_feature_array = _as_feature_array(clip_features, feature_type="clip", split=split)
    frequency_feature_array = _as_feature_array(frequency_features, feature_type="frequency", split=split)
    clip_label_array = _as_label_array(clip_labels, feature_type="clip", split=split)
    frequency_label_array = _as_label_array(frequency_labels, feature_type="frequency", split=split)
    clip_metadata = _as_meta(clip_meta, feature_type="clip", split=split)
    frequency_metadata = _as_meta(frequency_meta, feature_type="frequency", split=split)

    _validate_branch_rows(clip_feature_array, clip_label_array, clip_metadata, feature_type="clip", split=split)
    _validate_branch_rows(frequency_feature_array, frequency_label_array, frequency_metadata, feature_type="frequency", split=split)

    clip_ids = _image_ids(clip_metadata)
    frequency_ids = _image_ids(frequency_metadata)
    _validate_unique_ids(clip_ids, side="clip", split=split)
    _validate_unique_ids(frequency_ids, side="frequency", split=split)
    _validate_matching_id_sets(clip_ids, frequency_ids, split=split)

    frequency_index_by_id = {image_id: index for index, image_id in enumerate(frequency_ids)}
    frequency_order = np.asarray([frequency_index_by_id[image_id] for image_id in clip_ids], dtype=np.int64)
    aligned_frequency_features = frequency_feature_array[frequency_order]
    aligned_frequency_labels = frequency_label_array[frequency_order]
    aligned_frequency_meta = frequency_metadata.iloc[frequency_order].reset_index(drop=True)

    for row_index, image_id in enumerate(clip_ids):
        clip_label = int(clip_label_array[row_index])
        frequency_label = int(aligned_frequency_labels[row_index])
        if clip_label != frequency_label:
            raise NpyFeatureCacheError(
                f"fusion {split} label mismatch for image_id={image_id!r}: clip={clip_label} frequency={frequency_label}"
            )
        clip_meta_label = int(clip_metadata.iloc[row_index]["label"])
        frequency_meta_label = int(aligned_frequency_meta.iloc[row_index]["label"])
        if clip_meta_label != frequency_meta_label:
            raise NpyFeatureCacheError(
                f"fusion {split} metadata label mismatch for image_id={image_id!r}: clip={clip_meta_label} frequency={frequency_meta_label}"
            )

    fused = np.concatenate([clip_feature_array, aligned_frequency_features], axis=1).astype(np.float32, copy=False)
    return FusionFeatureTable(
        features=fused,
        labels=clip_label_array.astype(np.int64, copy=False),
        meta=clip_metadata.reset_index(drop=True),
        clip_features=clip_feature_array.astype(np.float32, copy=False),
        frequency_features=aligned_frequency_features.astype(np.float32, copy=False),
    )


def train_fusion(config: dict[str, object]) -> TrainingArtifacts:
    seed = _project_seed(config)
    set_seed(seed)
    device = torch.device(resolve_device(config))

    train_table = align_feature_tables(config, split="train")
    val_table = align_feature_tables(config, split="val")
    if train_table.features.shape[0] == 0:
        raise NpyFeatureCacheError("fusion train cache contains no rows")
    if val_table.features.shape[0] == 0:
        raise NpyFeatureCacheError("fusion val cache contains no rows")
    _validate_split_feature_dims(train_table, val_table)
    _validate_training_labels(train_table.labels, split="train")

    train_loader = _make_loader(train_table.features, train_table.labels, batch_size=_batch_size(config), shuffle=True, seed=seed)
    val_loader = _make_loader(val_table.features, val_table.labels, batch_size=_batch_size(config), shuffle=False, seed=seed)

    clip_dim = int(train_table.clip_features.shape[1])
    frequency_dim = int(train_table.frequency_features.shape[1])
    model = FusionClassifier(clip_dim=clip_dim, freq_dim=frequency_dim, hidden_dim=_hidden_dim(config), dropout=_dropout(config)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate(config), weight_decay=_weight_decay(config))

    checkpoint_path = _checkpoint_path(config, "fusion")
    train_log_path = _report_path(config, "fusion_train_log.csv")
    val_metrics_path = _report_path(config, "fusion_val_metrics.json")

    rows: list[dict[str, object]] = []
    best_score = (-math.inf, -math.inf)
    best_epoch = 0
    best_val_loss = math.inf
    best_val_roc_auc: float | None = None
    best_metrics: dict[str, object] = {}
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, _epochs(config) + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = _evaluate(model, val_loader, criterion, device, threshold=_threshold(config))
        score = _selection_score(val_loss, metrics["roc_auc"])
        rows.append(_log_row(epoch, train_loss, val_loss, metrics))

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
        raise RuntimeError("fusion training did not produce a validation checkpoint")

    input_dim = clip_dim + frequency_dim
    hidden_dim = _hidden_dim(config)
    _write_train_log(train_log_path, rows)
    _write_val_metrics(
        val_metrics_path,
        metrics=best_metrics,
        feature_type="fusion",
        model_name="FusionClassifier",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        threshold=_threshold(config),
        best_epoch=best_epoch,
    )
    save_checkpoint(
        checkpoint_path,
        model_state_dict=best_state,
        model_name="FusionClassifier",
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        threshold=_threshold(config),
        feature_type="fusion",
        config_snapshot=config,
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


def _validate_split_feature_dims(train_table: FusionFeatureTable, val_table: FusionFeatureTable) -> None:
    train_clip_dim = int(train_table.clip_features.shape[1])
    val_clip_dim = int(val_table.clip_features.shape[1])
    train_frequency_dim = int(train_table.frequency_features.shape[1])
    val_frequency_dim = int(val_table.frequency_features.shape[1])
    if train_clip_dim != val_clip_dim or train_frequency_dim != val_frequency_dim:
        message = (
            f"fusion train/val feature dimension mismatch: train clip={train_clip_dim} frequency={train_frequency_dim}; "
            f"val clip={val_clip_dim} frequency={val_frequency_dim}"
        )
        raise NpyFeatureCacheError(message)


def _load_required_branch(config: Mapping[str, object], *, feature_type: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    try:
        return load_feature_cache(config, feature_type=feature_type, split=split)
    except NpyFeatureCacheError as exc:
        paths = cache_paths(config, feature_type=feature_type, split=split)
        raise NpyFeatureCacheError(
            f"Fusion training requires {feature_type} cache for split={split!r}; expected features={paths.features} labels={paths.labels} meta={paths.meta}. {exc}"
        ) from exc


def _as_feature_array(value: np.ndarray | None, *, feature_type: str, split: str) -> np.ndarray:
    array = cast(np.ndarray, np.asarray(value, dtype=np.float32))
    if array.ndim != 2:
        raise NpyFeatureCacheError(f"fusion {split} {feature_type} features must be 2D, got shape {array.shape}")
    return array


def _as_label_array(value: np.ndarray | None, *, feature_type: str, split: str) -> np.ndarray:
    array = cast(np.ndarray, np.asarray(value, dtype=np.int64))
    if array.ndim != 1:
        raise NpyFeatureCacheError(f"fusion {split} {feature_type} labels must be 1D, got shape {array.shape}")
    return array


def _as_meta(value: pd.DataFrame | None, *, feature_type: str, split: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"fusion {split} {feature_type} metadata must be a pandas DataFrame")
    frame = value
    if "image_id" not in frame.columns or "label" not in frame.columns:
        raise NpyFeatureCacheError(f"fusion {split} {feature_type} metadata must include image_id and label columns")
    return frame.copy()


def _validate_branch_rows(features: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, *, feature_type: str, split: str) -> None:
    row_count = int(features.shape[0])
    if len(labels) != row_count:
        raise NpyFeatureCacheError(f"fusion {split} {feature_type} labels row count {len(labels)} does not match features row count {row_count}")
    if len(meta) != row_count:
        raise NpyFeatureCacheError(f"fusion {split} {feature_type} metadata row count {len(meta)} does not match features row count {row_count}")


def _image_ids(meta: pd.DataFrame) -> list[str]:
    return meta["image_id"].astype(str).tolist()


def _validate_unique_ids(image_ids: list[str], *, side: str, split: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for image_id in image_ids:
        if image_id in seen and image_id not in duplicates:
            duplicates.append(image_id)
        seen.add(image_id)
    if duplicates:
        raise NpyFeatureCacheError(f"fusion {split} {side} metadata image_id contains duplicate value(s): {duplicates[:5]}")


def _validate_matching_id_sets(clip_ids: list[str], frequency_ids: list[str], *, split: str) -> None:
    clip_set = set(clip_ids)
    frequency_set = set(frequency_ids)
    missing_from_frequency = sorted(clip_set - frequency_set)
    missing_from_clip = sorted(frequency_set - clip_set)
    if missing_from_frequency or missing_from_clip:
        parts: list[str] = []
        if missing_from_frequency:
            parts.append(f"missing from frequency: {missing_from_frequency[:10]}")
        if missing_from_clip:
            parts.append(f"missing from clip: {missing_from_clip[:10]}")
        raise NpyFeatureCacheError(f"fusion {split} image_id set mismatch; " + "; ".join(parts))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch fusion classifier on aligned CLIP and frequency .npy caches.")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    try:
        train_fusion(config)
    except NpyFeatureCacheError as exc:
        raise SystemExit(f"Fusion training failed clearly: {exc}") from None


if __name__ == "__main__":
    main()
