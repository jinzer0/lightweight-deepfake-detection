from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportArgumentType=false, reportUnusedCallResult=false, reportUnreachable=false

import csv
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score

from src.data.manifest import read_manifest, validate_manifest_rows
from src.features.cache import build_metadata, create_feature_cache, hash_manifest_file, write_feature_cache
from src.features.frequency import (
    DCT_BACKEND,
    DCT_POLICY,
    DEFAULT_FFT_EPSILON,
    DEFAULT_RADIAL_BINS,
    FEATURE_DTYPE,
    FrequencyFeatureConfig,
    extract_frequency_feature_batch,
)
from src.utils.image_io import DEFAULT_FREQUENCY_IMAGE_SIZE, load_rgb_image

RobustnessMode = Literal["quick", "full"]

ROBUSTNESS_COLUMNS = [
    "mode",
    "corruption_type",
    "corruption_level",
    "sample_count",
    "clean_accuracy",
    "corrupted_accuracy",
    "accuracy_degradation",
    "clean_f1_fake",
    "corrupted_f1_fake",
    "f1_fake_degradation",
    "clean_roc_auc",
    "corrupted_roc_auc",
    "roc_auc_degradation",
    "clean_average_precision",
    "corrupted_average_precision",
    "average_precision_degradation",
    "ranking_metric_input",
    "base_sample_ids",
    "corrupted_cache_path",
    "manifest_path",
    "experiment_dir",
    "model_path",
    "scaler_path",
]


@dataclass(frozen=True)
class RobustnessResult:
    output_dir: Path
    metrics_path: Path
    summary_path: Path
    rows: list[dict[str, Any]]


def robustness_levels(mode: RobustnessMode) -> list[tuple[str, str]]:
    if mode == "quick":
        return [("jpeg", "quality_75"), ("resize", "down_160"), ("blur", "sigma_1.0")]
    if mode == "full":
        return [
            ("jpeg", "quality_95"),
            ("jpeg", "quality_75"),
            ("jpeg", "quality_50"),
            ("jpeg", "quality_30"),
            ("resize", "down_160"),
            ("resize", "down_128"),
            ("blur", "sigma_0.5"),
            ("blur", "sigma_1.0"),
            ("blur", "sigma_2.0"),
        ]
    raise ValueError(f"unsupported robustness mode: {mode}")


def apply_corruption(image: Image.Image, corruption_type: str, corruption_level: str) -> Image.Image:
    rgb_image = image.convert("RGB")
    if corruption_type == "jpeg":
        quality = int(corruption_level.removeprefix("quality_"))
        buffer = io.BytesIO()
        _ = rgb_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            _ = compressed.load()
            return compressed.convert("RGB")
    if corruption_type == "resize":
        size = int(corruption_level.removeprefix("down_"))
        down = rgb_image.resize((size, size), resample=Image.Resampling.BICUBIC)
        return down.resize((DEFAULT_FREQUENCY_IMAGE_SIZE, DEFAULT_FREQUENCY_IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    if corruption_type == "blur":
        sigma = float(corruption_level.removeprefix("sigma_"))
        return rgb_image.filter(ImageFilter.GaussianBlur(radius=sigma))
    raise ValueError(f"unsupported corruption_type: {corruption_type}")


def run_frequency_robustness(
    *,
    manifest_path: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    mode: RobustnessMode = "quick",
    max_samples: int | None = None,
) -> RobustnessResult:
    manifest = Path(manifest_path)
    experiment = Path(experiment_dir)
    config = _read_yaml(experiment / "config.yaml")
    _require_frequency_artifact(config)

    rows = read_manifest(manifest)
    validate_manifest_rows(rows, strict=True)
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        raise ValueError("robustness requires at least one test row")
    if mode == "quick":
        test_rows = _tiny_balanced_subset(test_rows, max_samples or 4)
    elif max_samples is not None:
        test_rows = test_rows[: int(max_samples)]
    if not test_rows:
        raise ValueError("robustness selected no test rows")

    resolved_output = Path(output_dir) if output_dir is not None else Path("outputs") / "robustness" / experiment.name
    resolved_output.mkdir(parents=True, exist_ok=True)
    corruption_dir = resolved_output / "corrupted_images"
    cache_dir = resolved_output / "feature_caches"
    if corruption_dir.exists():
        shutil.rmtree(corruption_dir)
    corruption_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = experiment / "model.joblib"
    scaler_path = experiment / "scaler.joblib"
    model = joblib.load(model_path)
    transformers = joblib.load(scaler_path)
    scaler = transformers.get("frequency_scaler") if isinstance(transformers, dict) else None
    if scaler is None:
        raise ValueError("scaler.joblib must contain frequency_scaler for frequency robustness")

    threshold = float(config.get("threshold", 0.5))
    clean_features = _scale_frequency_features(_extract_raw_frequency(test_rows), scaler)
    labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    clean_prediction = _predict(model, clean_features, threshold=threshold)
    clean_metrics = _metrics(labels, clean_prediction)

    result_rows: list[dict[str, Any]] = []
    base_sample_ids = [str(row.get("base_sample_id") or row.get("sample_id")) for row in test_rows]
    clean_manifest_hash = hash_manifest_file(manifest)
    for corruption_type, corruption_level in robustness_levels(mode):
        corrupted_paths = _write_corrupted_images(test_rows, corruption_dir, corruption_type, corruption_level)
        corrupted_rows = [dict(row, root="", rel_path=path.as_posix()) for row, path in zip(test_rows, corrupted_paths, strict=True)]
        raw_corrupted_features = _extract_raw_frequency(corrupted_rows)
        corrupted_cache_path = _write_corrupted_frequency_cache(
            corrupted_rows,
            raw_corrupted_features,
            cache_dir=cache_dir,
            base_sample_ids=base_sample_ids,
            corruption_type=corruption_type,
            corruption_level=corruption_level,
            clean_manifest_hash=clean_manifest_hash,
            seed=int(config.get("seed", 42)),
        )
        scaled_corrupted_features = _scale_frequency_features(raw_corrupted_features, scaler)
        corrupted_prediction = _predict(model, scaled_corrupted_features, threshold=threshold)
        corrupted_metrics = _metrics(labels, corrupted_prediction)
        result_rows.append(
            _result_row(
                mode=mode,
                corruption_type=corruption_type,
                corruption_level=corruption_level,
                sample_count=len(test_rows),
                clean_metrics=clean_metrics,
                corrupted_metrics=corrupted_metrics,
                base_sample_ids=base_sample_ids,
                corrupted_cache_path=corrupted_cache_path,
                manifest_path=manifest,
                experiment_dir=experiment,
                model_path=model_path,
                scaler_path=scaler_path,
            )
        )

    metrics_path = resolved_output / "robustness_metrics.csv"
    summary_path = resolved_output / "robustness_summary.png"
    _write_metrics_csv(metrics_path, result_rows)
    _write_summary_plot(summary_path, result_rows)
    return RobustnessResult(resolved_output, metrics_path, summary_path, result_rows)


def _require_frequency_artifact(config: dict[str, Any]) -> None:
    mode = str(config.get("mode", ""))
    if mode != "frequency_only":
        raise ValueError(f"robustness currently supports frequency_only artifacts only; got mode={mode!r}")
    classifier_key = str(config.get("classifier", {}).get("key", ""))
    if classifier_key not in {"logistic_regression", "linear_svm"}:
        raise ValueError(f"unsupported classifier for robustness: {classifier_key!r}")


def _tiny_balanced_subset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in ["0", "1"]:
        selected.extend(row for row in rows if row.get("label") == label)
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in selected + rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in seen:
            unique.append(row)
            seen.add(sample_id)
        if len(unique) >= int(limit):
            break
    return unique


def _extract_raw_frequency(rows: list[dict[str, str]]) -> np.ndarray:
    paths = [_absolute_image_path(row) for row in rows]
    config = FrequencyFeatureConfig()
    features = extract_frequency_feature_batch(
        paths,
        image_size=config.image_size,
        radial_bins=config.radial_bins,
        fft_epsilon=config.fft_epsilon,
    )
    raw_features = np.asarray(features, dtype=FEATURE_DTYPE)
    if raw_features.ndim != 2 or not np.isfinite(raw_features).all():
        raise ValueError("raw frequency features must be finite 2D array")
    return raw_features


def _scale_frequency_features(raw_features: np.ndarray, scaler: Any) -> np.ndarray:
    transformed = scaler.transform(raw_features)
    features = np.asarray(transformed, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("scaled frequency features must be finite 2D array")
    return features


def _write_corrupted_frequency_cache(
    rows: list[dict[str, str]],
    features: np.ndarray,
    *,
    cache_dir: Path,
    base_sample_ids: list[str],
    corruption_type: str,
    corruption_level: str,
    clean_manifest_hash: str,
    seed: int,
) -> Path:
    feature_config = FrequencyFeatureConfig().as_dict()
    metadata = build_metadata(
        feature_dim=int(features.shape[1]),
        dtype=str(np.dtype(FEATURE_DTYPE).name),
        normalization="raw_unscaled",
        seed=int(seed),
        extra={
            "image_size": DEFAULT_FREQUENCY_IMAGE_SIZE,
            "radial_bins": DEFAULT_RADIAL_BINS,
            "fft_epsilon": DEFAULT_FFT_EPSILON,
            "dct_policy": DCT_POLICY,
            "dct_backend": DCT_BACKEND,
            "is_corrupted": True,
            "base_sample_ids": list(base_sample_ids),
            "corruption_type": corruption_type,
            "corruption_level": corruption_level,
            "clean_manifest_hash": clean_manifest_hash,
        },
    )
    cache = create_feature_cache(
        manifest_rows=rows,
        feature_type="frequency",
        feature_config=feature_config,
        features=features,
        metadata=metadata,
    )
    cache_path = cache_dir / f"frequency_{corruption_type}_{corruption_level}.pt"
    write_feature_cache(cache, cache_path)
    return cache_path


def _absolute_image_path(row: dict[str, str]) -> Path:
    rel_path = Path(str(row["rel_path"]))
    if rel_path.is_absolute():
        return rel_path
    root = str(row.get("root", ""))
    return Path(root) / rel_path if root else rel_path


def _write_corrupted_images(
    rows: list[dict[str, str]], output_root: Path, corruption_type: str, corruption_level: str
) -> list[Path]:
    level_dir = output_root / corruption_type / corruption_level
    level_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, row in enumerate(rows):
        image = load_rgb_image(_absolute_image_path(row))
        corrupted = apply_corruption(image, corruption_type, corruption_level)
        output_path = level_dir / f"{index:05d}_{row['sample_id']}.png"
        corrupted.save(output_path, format="PNG")
        paths.append(output_path)
    return paths


def _predict(model: Any, features: np.ndarray, *, threshold: float) -> dict[str, np.ndarray | str]:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        classes = np.asarray(model.classes_, dtype=np.int64)
        matching = np.flatnonzero(classes == 1)
        if matching.size != 1:
            raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
        prob_fake = probabilities[:, int(matching[0])]
        pred_label = (prob_fake >= float(threshold)).astype(np.int64)
        return {"pred_label": pred_label, "ranking": prob_fake, "ranking_metric_input": "prob_fake"}
    decision = np.asarray(model.decision_function(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if decision.ndim != 1:
        matching = np.flatnonzero(classes == 1)
        if matching.size != 1:
            raise ValueError(f"model classes must contain label 1 exactly once, got {classes.tolist()}")
        score = decision[:, int(matching[0])]
    else:
        score = decision if int(classes[1]) == 1 else -decision
    return {"pred_label": (score >= 0.0).astype(np.int64), "ranking": score, "ranking_metric_input": "decision_score"}


def _metrics(labels: np.ndarray, prediction: dict[str, np.ndarray | str]) -> dict[str, Any]:
    pred_label = np.asarray(prediction["pred_label"], dtype=np.int64)
    ranking = np.asarray(prediction["ranking"], dtype=np.float64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_label, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, pred_label)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "ranking_metric_input": str(prediction["ranking_metric_input"]),
    }
    if set(np.unique(labels).astype(int).tolist()) == {0, 1}:
        metrics["roc_auc"] = float(roc_auc_score(labels, ranking))
        metrics["average_precision"] = float(average_precision_score(labels, ranking))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def _result_row(
    *,
    mode: str,
    corruption_type: str,
    corruption_level: str,
    sample_count: int,
    clean_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    base_sample_ids: list[str],
    corrupted_cache_path: Path,
    manifest_path: Path,
    experiment_dir: Path,
    model_path: Path,
    scaler_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": mode,
        "corruption_type": corruption_type,
        "corruption_level": corruption_level,
        "sample_count": int(sample_count),
        "ranking_metric_input": clean_metrics["ranking_metric_input"],
        "base_sample_ids": json.dumps(base_sample_ids),
        "corrupted_cache_path": corrupted_cache_path.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "experiment_dir": experiment_dir.as_posix(),
        "model_path": model_path.as_posix(),
        "scaler_path": scaler_path.as_posix(),
    }
    for metric_name in ["accuracy", "f1_fake", "roc_auc", "average_precision"]:
        clean_value = clean_metrics.get(metric_name)
        corrupted_value = corrupted_metrics.get(metric_name)
        prefix = "f1_fake" if metric_name == "f1_fake" else metric_name
        row[f"clean_{prefix}"] = clean_value
        row[f"corrupted_{prefix}"] = corrupted_value
        row[f"{prefix}_degradation"] = None if clean_value is None or corrupted_value is None else float(clean_value) - float(corrupted_value)
    return row


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROBUSTNESS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in ROBUSTNESS_COLUMNS})


def _write_summary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['corruption_type']}\n{row['corruption_level']}" for row in rows]
    accuracy = [float(row["accuracy_degradation"] or 0.0) for row in rows]
    f1_values = [float(row["f1_fake_degradation"] or 0.0) for row in rows]
    x_values = np.arange(len(rows))
    width = 0.35
    fig, axis = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4))
    _ = axis.bar(x_values - width / 2, accuracy, width, label="Accuracy degradation")
    _ = axis.bar(x_values + width / 2, f1_values, width, label="F1(fake) degradation")
    _ = axis.set_ylabel("Clean - corrupted")
    _ = axis.set_title("Robustness degradation by corruption")
    _ = axis.set_xticks(x_values, labels, rotation=30, ha="right")
    _ = axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML object at {path}")
    return data


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)
