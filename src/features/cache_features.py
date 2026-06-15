from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ALLOWED_SPLITS, ImageMetadataDataset
from src.data.transforms import get_eval_transform
from src.features.clip_features import ClipModelLoadError, extract_clip_features, load_clip_model_and_preprocess
from src.features.frequency_features import FEATURE_DTYPE, extract_frequency_feature
from src.utils.config import load_config, resolve_device

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is a progress nicety
    tqdm = None


FEATURE_TYPES = ("clip", "frequency")
REQUIRED_META_COLUMNS = ("image_id", "filepath", "label", "class_name", "dataset", "generator", "split", "width", "height", "ext")
T = TypeVar("T")


class NpyFeatureCacheError(ValueError):
    pass


def cache_split(config: Mapping[str, Any], feature_type: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    _validate_feature_type(feature_type)
    _validate_split(split)
    if feature_type == "frequency":
        return cache_frequency_split(config, split)
    return cache_clip_split(config, split)


def cache_frequency_split(config: Mapping[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    _validate_split(split)
    dataset_csv = _dataset_csv(config)
    dataset = ImageMetadataDataset(dataset_csv, split=split, transform=None, return_metadata=True)
    meta = _metadata_for_csv(dataset_csv, dataset.rows, split=split)

    features: list[np.ndarray] = []
    labels: list[int] = []
    for index in _progress(range(len(dataset)), desc=f"Caching frequency {split}", total=len(dataset), unit="image"):
        image, label, _metadata = cast(tuple[Any, int, dict[str, str]], dataset[index])
        features.append(extract_frequency_feature(image, dict(config)))
        labels.append(int(label))

    feature_array = _stack_features(features, dtype=np.dtype(FEATURE_DTYPE))
    label_array = np.asarray(labels, dtype=np.int64)
    return write_feature_cache(config, feature_type="frequency", split=split, features=feature_array, labels=label_array, meta=meta)


def cache_clip_split(config: Mapping[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    _validate_split(split)
    dataset_csv = _dataset_csv(config)
    device = resolve_device(dict(config))
    try:
        model, preprocess = load_clip_model_and_preprocess(config, device=device)
    except ClipModelLoadError:
        raise
    except Exception as exc:
        raise ClipModelLoadError(f"Failed optional CLIP cache smoke for split={split!r}: {exc}") from exc
    dataset = ImageMetadataDataset(dataset_csv, split=split, transform=preprocess, return_metadata=True)
    dataloader = DataLoader(
        dataset,
        batch_size=_batch_size(config),
        shuffle=False,
        num_workers=_num_workers(config),
    )
    features, labels, extracted_meta = extract_clip_features(
        model,
        _progress(dataloader, desc=f"Caching CLIP {split}", total=len(dataloader), unit="batch"),
        device=device,
        normalize=_clip_normalize(config),
    )
    source_meta = _metadata_for_csv(dataset_csv, dataset.rows, split=split)
    meta = _align_clip_metadata(source_meta, extracted_meta, split=split)
    return write_feature_cache(config, feature_type="clip", split=split, features=features, labels=labels, meta=meta)


def write_feature_cache(
    config: Mapping[str, Any],
    *,
    feature_type: str,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    paths = cache_paths(config, feature_type=feature_type, split=split)
    paths.feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.features, np.asarray(features, dtype=np.float32))
    np.save(paths.labels, np.asarray(labels, dtype=np.int64))
    meta.to_csv(paths.meta, index=False)

    loaded = load_feature_cache(config, feature_type=feature_type, split=split)
    print(f"Saved {feature_type} {split} features: {paths.features}")
    print(f"features shape: {loaded[0].shape}")
    print(f"labels shape: {loaded[1].shape}")
    print(f"meta shape: {loaded[2].shape}")
    return loaded


def load_feature_cache(config: Mapping[str, Any], *, feature_type: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    paths = cache_paths(config, feature_type=feature_type, split=split)
    missing = [path for path in (paths.features, paths.labels, paths.meta) if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise NpyFeatureCacheError(f"Missing {feature_type} {split} cache file(s): {missing_text}")

    features = np.load(paths.features)
    labels = np.load(paths.labels)
    meta = pd.read_csv(paths.meta, dtype={"image_id": str, "filepath": str, "label": "Int64", "split": str})
    validate_feature_cache_arrays(features, labels, meta, feature_type=feature_type, split=split, context=str(paths.feature_dir))
    return features, labels.astype(np.int64, copy=False), meta


def validate_feature_cache_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    *,
    feature_type: str,
    split: str,
    context: str = "feature cache",
) -> None:
    _validate_feature_type(feature_type)
    _validate_split(split)
    errors: list[str] = []

    if features.ndim != 2:
        errors.append(f"{context}: {feature_type} {split} features must be 2D, got shape {features.shape}")
    if labels.ndim != 1:
        errors.append(f"{context}: {feature_type} {split} labels must be 1D, got shape {labels.shape}")

    row_count = int(features.shape[0]) if features.ndim >= 1 else 0
    label_count = int(labels.shape[0]) if labels.ndim >= 1 else 0
    if label_count != row_count:
        errors.append(f"{context}: {feature_type} {split} labels row count {label_count} does not match features row count {row_count}")
    if len(meta) != row_count:
        errors.append(f"{context}: {feature_type} {split} metadata row count {len(meta)} does not match features row count {row_count}")

    missing_columns = [column for column in REQUIRED_META_COLUMNS if column not in meta.columns]
    if missing_columns:
        errors.append(f"{context}: {feature_type} {split} metadata missing required column(s): {', '.join(missing_columns)}")
    else:
        _validate_metadata_values(meta, labels, feature_type=feature_type, split=split, context=context, errors=errors)

    if features.ndim == 2 and not np.isfinite(features).all():
        errors.append(f"{context}: {feature_type} {split} features contain non-finite values")

    if errors:
        raise NpyFeatureCacheError("; ".join(errors))


class CachePaths(NamedTuple):
    feature_dir: Path
    features: Path
    labels: Path
    meta: Path


def cache_paths(config: Mapping[str, Any], *, feature_type: str, split: str) -> CachePaths:
    _validate_feature_type(feature_type)
    _validate_split(split)
    feature_dir = _feature_dir(config) / feature_type
    return CachePaths(feature_dir, feature_dir / f"{split}_features.npy", feature_dir / f"{split}_labels.npy", feature_dir / f"{split}_meta.csv")


def _metadata_for_dataset_rows(rows: Sequence[Mapping[str, Any]], *, split: str, context: str) -> pd.DataFrame:
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        frame = pd.DataFrame({column: [] for column in REQUIRED_META_COLUMNS})
    for column in REQUIRED_META_COLUMNS:
        if column not in frame.columns:
            raise NpyFeatureCacheError(f"{context}: split={split!r} dataset metadata missing required column {column!r}")
    frame = frame.loc[:, list(REQUIRED_META_COLUMNS)].copy()
    frame["label"] = frame["label"].astype(int)
    validate_feature_cache_arrays(
        np.empty((len(frame), 0), dtype=np.float32),
        frame["label"].to_numpy(dtype=np.int64),
        frame,
        feature_type="frequency",
        split=split,
        context=context,
    )
    return frame


def _metadata_for_csv(csv_path: Path, dataset_rows: Sequence[Mapping[str, Any]], *, split: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, dtype=str)
    split_frame = frame.loc[frame["split"].astype(str) == split].copy()
    meta = _metadata_for_dataset_rows(split_frame.to_dict(orient="records"), split=split, context=str(csv_path))
    dataset_ids = [str(row["image_id"]) for row in dataset_rows]
    meta_ids = meta["image_id"].astype(str).tolist()
    if meta_ids != dataset_ids:
        raise NpyFeatureCacheError(f"{csv_path}: split={split!r} dataset.csv row order does not match ImageMetadataDataset row order")
    return meta


def _align_clip_metadata(source_meta: pd.DataFrame, extracted_meta: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if not extracted_meta.empty and "image_id" in extracted_meta.columns:
        extracted_ids = extracted_meta["image_id"].astype(str).tolist()
        source_ids = source_meta["image_id"].astype(str).tolist()
        if extracted_ids != source_ids:
            raise NpyFeatureCacheError(f"CLIP {split} metadata image_id order does not match dataset.csv")
    return source_meta


def _validate_metadata_values(
    meta: pd.DataFrame,
    labels: np.ndarray,
    *,
    feature_type: str,
    split: str,
    context: str,
    errors: list[str],
) -> None:
    image_ids = cast(Any, meta["image_id"].astype(str))
    duplicates = cast(list[str], image_ids[image_ids.duplicated()].unique().tolist())
    if duplicates:
        errors.append(f"{context}: {feature_type} {split} metadata image_id contains duplicate value(s): {duplicates[:5]}")

    meta_labels = cast(Any, pd.to_numeric(meta["label"], errors="coerce"))
    if bool(meta_labels.isna().any()):
        errors.append(f"{context}: {feature_type} {split} metadata label contains non-integer value(s)")
    elif labels.ndim == 1 and len(labels) == len(meta):
        expected = cast(np.ndarray, meta_labels.astype(np.int64).to_numpy())
        actual = labels.astype(np.int64, copy=False)
        if not np.array_equal(actual, expected):
            mismatch_indices = np.flatnonzero(actual != expected)
            first = int(mismatch_indices[0]) if len(mismatch_indices) else -1
            image_id = str(meta.iloc[first]["image_id"]) if first >= 0 else "unknown"
            errors.append(
                f"{context}: {feature_type} {split} labels do not match metadata label at row {first} image_id={image_id!r} "
                f"labels={int(actual[first]) if first >= 0 else 'n/a'} metadata={int(expected[first]) if first >= 0 else 'n/a'}"
            )

    split_values = set(meta["split"].astype(str).tolist())
    if split_values != {split}:
        errors.append(f"{context}: {feature_type} metadata split values {sorted(split_values)} do not match requested split {split!r}")


def _stack_features(features: Sequence[np.ndarray], *, dtype: np.dtype[Any]) -> np.ndarray:
    if not features:
        return np.empty((0, 0), dtype=dtype)
    return np.stack(features, axis=0).astype(dtype, copy=False)


def _dataset_csv(config: Mapping[str, Any]) -> Path:
    paths = _paths_config(config)
    return Path(str(paths["dataset_csv"]))


def _feature_dir(config: Mapping[str, Any]) -> Path:
    paths = _paths_config(config)
    return Path(str(paths["feature_dir"]))


def _paths_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise NpyFeatureCacheError("config must contain a paths mapping")
    for key in ("dataset_csv", "feature_dir"):
        if key not in paths:
            raise NpyFeatureCacheError(f"config.paths missing required key {key!r}")
    return paths


def _image_size(config: Mapping[str, Any]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "image_size" in data:
        return int(cast(Any, data["image_size"]))
    return 224


def _batch_size(config: Mapping[str, Any]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "batch_size" in data:
        return int(cast(Any, data["batch_size"]))
    return 32


def _num_workers(config: Mapping[str, Any]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "num_workers" in data:
        return int(cast(Any, data["num_workers"]))
    return 0


def _clip_normalize(config: Mapping[str, Any]) -> bool:
    clip = config.get("clip")
    if isinstance(clip, Mapping):
        return bool(clip.get("normalize_feature", True))
    return True


def _progress(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)


def _validate_feature_type(feature_type: str) -> None:
    if feature_type not in FEATURE_TYPES:
        raise NpyFeatureCacheError(f"Invalid feature_type {feature_type!r}; expected one of: {', '.join(FEATURE_TYPES)}")


def _validate_split(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise NpyFeatureCacheError(f"Invalid split {split!r}; expected one of: {', '.join(ALLOWED_SPLITS)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write validated .npy feature caches from canonical dataset.csv metadata.")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    parser.add_argument("--feature_type", required=True, choices=FEATURE_TYPES, help="Feature cache type to write")
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS, help="Dataset split to cache")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        config = load_config(args.config)
        cache_split(config, feature_type=args.feature_type, split=args.split)
    except ClipModelLoadError as exc:
        raise SystemExit(f"CLIP cache optional-smoke failed clearly: {exc}") from None
    except (FileNotFoundError, NpyFeatureCacheError, ValueError) as exc:
        raise SystemExit(f"Feature cache failed clearly: {exc}") from None


if __name__ == "__main__":
    main()
