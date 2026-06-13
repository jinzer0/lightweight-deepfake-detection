from __future__ import annotations

# pyright: reportAny=false, reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportExplicitAny=false, reportUnusedCallResult=false

import argparse
import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, override

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ALLOWED_SPLITS
from src.data.transforms import get_eval_transform
from src.data.validate_metadata import MetadataValidationError, read_metadata, validate_metadata
from src.eval.evaluate import MODEL_SPECS
from src.eval.metrics import compute_binary_metrics
from src.features.clip_features import ClipModelLoadError, extract_clip_features, load_clip_model
from src.features.frequency_features import FEATURE_DTYPE, extract_frequency_feature
from src.models.checkpoint import CheckpointError, load_checkpoint
from src.models.fusion_classifier import FusionClassifier
from src.models.mlp_classifier import MLPClassifier
from src.utils.config import load_config, resolve_device
from src.utils.image_io import load_rgb_image


ModelName = Literal["clip_only", "frequency_only", "fusion"]

ROBUSTNESS_COLUMNS = ["model_name", "corruption", "severity", "accuracy", "precision", "recall", "f1", "roc_auc"]


class RobustnessError(ValueError):
    pass


@dataclass(frozen=True)
class CorruptionSpec:
    corruption: str
    severity: str


@dataclass(frozen=True)
class RobustnessResult:
    metrics_path: Path
    rows: list[dict[str, object]]


class _InMemoryImageDataset(Dataset[tuple[torch.Tensor, int, dict[str, str]]]):
    def __init__(self, images: Sequence[Image.Image], labels: Sequence[int], rows: Sequence[Mapping[str, str]], image_size: int) -> None:
        if len(images) != len(labels) or len(images) != len(rows):
            raise RobustnessError("corrupted image, label, and metadata row counts must match")
        self.images: list[Image.Image] = list(images)
        self.labels: list[int] = [int(label) for label in labels]
        self.rows: list[dict[str, str]] = [dict(row) for row in rows]
        self.transform: Any = get_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.images)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, str]]:
        return self.transform(self.images[index]), self.labels[index], self.rows[index]


def default_corruptions(config: Mapping[str, object]) -> list[CorruptionSpec]:
    robustness = config.get("robustness")
    settings = robustness if isinstance(robustness, Mapping) else {}
    jpeg_qualities = settings.get("jpeg_qualities", [95, 75, 50])
    resize_scales = settings.get("resize_scales", [0.5])
    blur_sigmas = settings.get("blur_sigmas", [1.0, 2.0])

    specs: list[CorruptionSpec] = []
    specs.extend(CorruptionSpec("jpeg", str(int(quality))) for quality in cast(Sequence[Any], jpeg_qualities))
    specs.extend(CorruptionSpec("resize", _severity_text(float(scale))) for scale in cast(Sequence[Any], resize_scales))
    specs.extend(CorruptionSpec("blur", _severity_text(float(sigma))) for sigma in cast(Sequence[Any], blur_sigmas))
    return specs


def apply_corruption(image: Image.Image, corruption: str, severity: str) -> Image.Image:
    rgb_image = image.convert("RGB")
    if corruption == "jpeg":
        quality = _severity_int(severity, prefix="quality_")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            return compressed.convert("RGB")
    if corruption == "resize":
        scale = _resize_scale(severity)
        width, height = rgb_image.size
        down_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        down = rgb_image.resize(down_size, resample=Image.Resampling.BICUBIC)
        return down.resize((width, height), resample=Image.Resampling.BICUBIC)
    if corruption == "blur":
        sigma = _severity_float(severity, prefix="sigma_")
        return rgb_image.filter(ImageFilter.GaussianBlur(radius=sigma))
    raise RobustnessError(f"unsupported corruption: {corruption}")


def evaluate_robustness(config: Mapping[str, object], *, model_name: ModelName, split: str) -> RobustnessResult:
    if model_name not in MODEL_SPECS:
        raise RobustnessError(f"model must be one of {sorted(MODEL_SPECS)}, got {model_name!r}")
    if split not in ALLOWED_SPLITS:
        raise RobustnessError(f"split must be one of {', '.join(ALLOWED_SPLITS)}, got {split!r}")

    rows = _load_split_rows(config, split=split)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    checkpoint = _load_model_checkpoint(config, model_name=model_name)
    threshold = _checkpoint_float(checkpoint["threshold"], "threshold")
    device = torch.device(resolve_device(dict(config)))

    result_rows: list[dict[str, object]] = []
    for spec in default_corruptions(config):
        corrupted_images = [_corrupted_row_image(row, spec) for row in rows]
        table = _extract_corrupted_features(config, model_name=model_name, rows=rows, labels=labels, images=corrupted_images, device=device)
        model = _build_model(checkpoint, model_name=model_name, features=table.features, clip_dim=table.clip_dim, frequency_dim=table.frequency_dim)
        probabilities = _predict_probabilities(model, table.features, device=device)
        metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
        result_rows.append(
            {
                "model_name": model_name,
                "corruption": spec.corruption,
                "severity": spec.severity,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    report_dir = _report_dir(config)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / f"{model_name}_robustness_metrics.csv"
    _write_metrics_csv(metrics_path, result_rows)
    return RobustnessResult(metrics_path=metrics_path, rows=result_rows)


def run_frequency_robustness(
    *,
    manifest_path: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    mode: str = "quick",
    max_samples: int | None = None,
) -> RobustnessResult:
    del experiment_dir, mode, max_samples
    config: dict[str, object] = {
        "project": {"device": "cpu"},
        "paths": {
            "dataset_csv": str(manifest_path),
            "checkpoint_dir": "artifacts/checkpoints",
            "report_dir": str(output_dir or Path("artifacts") / "reports"),
        },
        "frequency": {"method": "dct", "image_size": 224, "radial_bins": 64, "log_scale": True, "normalize_feature": True},
        "robustness": {"jpeg_qualities": [75], "resize_scales": [0.5], "blur_sigmas": [1.0]},
    }
    return evaluate_robustness(config, model_name="frequency_only", split="test")


@dataclass(frozen=True)
class _FeatureTable:
    features: np.ndarray
    clip_dim: int | None = None
    frequency_dim: int | None = None


def _load_split_rows(config: Mapping[str, object], *, split: str) -> list[dict[str, str]]:
    dataset_csv = _dataset_csv(config)
    try:
        validate_metadata(dataset_csv, strict=True)
    except MetadataValidationError as exc:
        raise RobustnessError(f"Invalid dataset.csv for robustness: {exc}") from exc
    rows = [row for row in read_metadata(dataset_csv) if row.get("split") == split]
    if not rows:
        raise RobustnessError(f"dataset.csv has no rows for split={split!r}")
    return rows


def _corrupted_row_image(row: Mapping[str, str], spec: CorruptionSpec) -> Image.Image:
    image = load_rgb_image(row["filepath"])
    return apply_corruption(image, spec.corruption, spec.severity)


def _extract_corrupted_features(
    config: Mapping[str, object],
    *,
    model_name: ModelName,
    rows: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    images: Sequence[Image.Image],
    device: torch.device,
) -> _FeatureTable:
    frequency_features: np.ndarray | None = None
    clip_features: np.ndarray | None = None
    if model_name in {"frequency_only", "fusion"}:
        frequency_features = _extract_frequency_features(config, images)
    if model_name in {"clip_only", "fusion"}:
        clip_features = _extract_clip_feature_table(config, rows=rows, labels=labels, images=images, device=device)
    if model_name == "frequency_only":
        assert frequency_features is not None
        return _FeatureTable(features=frequency_features, frequency_dim=int(frequency_features.shape[1]))
    if model_name == "clip_only":
        assert clip_features is not None
        return _FeatureTable(features=clip_features, clip_dim=int(clip_features.shape[1]))
    if frequency_features is None or clip_features is None:
        raise RobustnessError("fusion robustness requires both CLIP and frequency features")
    features = np.concatenate([clip_features, frequency_features], axis=1).astype(np.float32, copy=False)
    return _FeatureTable(features=features, clip_dim=int(clip_features.shape[1]), frequency_dim=int(frequency_features.shape[1]))


def _extract_frequency_features(config: Mapping[str, object], images: Sequence[Image.Image]) -> np.ndarray:
    features = [extract_frequency_feature(image, config) for image in images]
    if not features:
        raise RobustnessError("frequency robustness selected no images")
    array = np.stack(features, axis=0).astype(FEATURE_DTYPE, copy=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise RobustnessError(f"frequency features must be finite 2D array, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def _extract_clip_feature_table(
    config: Mapping[str, object], *, rows: Sequence[Mapping[str, str]], labels: np.ndarray, images: Sequence[Image.Image], device: torch.device
) -> np.ndarray:
    try:
        model = load_clip_model(config, device=device)
    except ClipModelLoadError:
        raise
    except Exception as exc:
        raise ClipModelLoadError(f"Failed optional CLIP robustness feature extraction: {exc}") from exc
    dataset = _InMemoryImageDataset(images, labels.tolist(), rows, _image_size(config))
    dataloader = DataLoader(dataset, batch_size=_batch_size(config), shuffle=False, num_workers=0)
    features, extracted_labels, _meta = extract_clip_features(model, dataloader, device=device, normalize=_clip_normalize(config))
    if not np.array_equal(extracted_labels.astype(np.int64, copy=False), labels):
        raise RobustnessError("CLIP robustness labels do not match dataset.csv labels")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise RobustnessError(f"CLIP features must be finite 2D array, got shape {features.shape}")
    return features.astype(np.float32, copy=False)


def _load_model_checkpoint(config: Mapping[str, object], *, model_name: ModelName) -> dict[str, object]:
    spec = MODEL_SPECS[model_name]
    checkpoint_path = _checkpoint_dir(config) / str(spec["checkpoint"])
    if not checkpoint_path.is_file():
        raise RobustnessError(f"Missing checkpoint file: {checkpoint_path}")
    return load_checkpoint(checkpoint_path, expected_feature_type=str(spec["feature_type"]))


def _build_model(
    checkpoint: Mapping[str, object], *, model_name: ModelName, features: np.ndarray, clip_dim: int | None, frequency_dim: int | None
) -> nn.Module:
    input_dim = _checkpoint_int(checkpoint["input_dim"], "input_dim")
    hidden_dim = _checkpoint_int(checkpoint["hidden_dim"], "hidden_dim")
    if input_dim != int(features.shape[1]):
        raise CheckpointError(f"checkpoint input_dim {input_dim} does not match robustness feature dimension {features.shape[1]}")
    checkpoint_model_name = str(checkpoint["model_name"])
    if model_name == "fusion":
        if checkpoint_model_name != "FusionClassifier":
            raise CheckpointError(f"fusion checkpoint model_name must be FusionClassifier, got {checkpoint_model_name!r}")
        if clip_dim is None or frequency_dim is None:
            raise CheckpointError("fusion robustness requires CLIP and frequency feature dimensions")
        model: nn.Module = FusionClassifier(clip_dim=clip_dim, freq_dim=frequency_dim, hidden_dim=hidden_dim, dropout=0.0)
    else:
        if checkpoint_model_name != "MLPClassifier":
            raise CheckpointError(f"{model_name} checkpoint model_name must be MLPClassifier, got {checkpoint_model_name!r}")
        model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state = cast(Mapping[str, torch.Tensor], checkpoint["model_state_dict"])
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_probabilities(model: nn.Module, features: np.ndarray, *, device: torch.device) -> np.ndarray:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64, copy=False)
    if probabilities.ndim != 1 or int(probabilities.shape[0]) != int(features.shape[0]):
        raise RobustnessError(f"model must return one fake-probability per row, got shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise RobustnessError("model produced invalid fake probabilities")
    return probabilities


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROBUSTNESS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in ROBUSTNESS_COLUMNS})


def _dataset_csv(config: Mapping[str, object]) -> Path:
    paths = _paths(config)
    if "dataset_csv" not in paths:
        raise RobustnessError("config.paths missing required key 'dataset_csv'")
    return Path(str(paths["dataset_csv"]))


def _checkpoint_dir(config: Mapping[str, object]) -> Path:
    return Path(str(_paths(config).get("checkpoint_dir", "artifacts/checkpoints")))


def _report_dir(config: Mapping[str, object]) -> Path:
    return Path(str(_paths(config).get("report_dir", "artifacts/reports")))


def _paths(config: Mapping[str, object]) -> Mapping[str, object]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise RobustnessError("config.paths must be a mapping")
    return paths


def _image_size(config: Mapping[str, object]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "image_size" in data:
        return int(cast(Any, data["image_size"]))
    return 224


def _batch_size(config: Mapping[str, object]) -> int:
    data = config.get("data")
    if isinstance(data, Mapping) and "batch_size" in data:
        return int(cast(Any, data["batch_size"]))
    return 32


def _clip_normalize(config: Mapping[str, object]) -> bool:
    clip = config.get("clip")
    if isinstance(clip, Mapping):
        return bool(clip.get("normalize_feature", True))
    return True


def _checkpoint_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be an int")
    return value


def _checkpoint_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CheckpointError(f"checkpoint {name} must be numeric")
    return float(value)


def _severity_text(value: float) -> str:
    return str(float(value))


def _severity_int(severity: str, *, prefix: str) -> int:
    text = severity.removeprefix(prefix)
    return int(float(text))


def _severity_float(severity: str, *, prefix: str) -> float:
    return float(severity.removeprefix(prefix))


def _resize_scale(severity: str) -> float:
    if severity.startswith("down_"):
        pixels = float(severity.removeprefix("down_"))
        return pixels / 224.0
    return _severity_float(severity, prefix="scale_") if severity.startswith("scale_") else float(severity)


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint robustness on corrupted dataset.csv images.")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS), help="Model checkpoint to evaluate")
    parser.add_argument("--split", default="test", choices=ALLOWED_SPLITS, help="dataset.csv split to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    try:
        result = evaluate_robustness(config, model_name=cast(ModelName, args.model), split=str(args.split))
    except (RobustnessError, CheckpointError, ClipModelLoadError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Robustness evaluation failed clearly: {exc}") from None
    print(f"model_name={args.model}")
    print(f"split={args.split}")
    print(f"corruptions={len(result.rows)}")
    print(f"saved metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
