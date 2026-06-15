from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportArgumentType=false

import argparse
import csv
import io
import json
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - tqdm is a progress nicety
    tqdm = None

from src.data.transforms import get_eval_transform
from src.features.clip_features import ClipModelLoadError, load_clip_model_and_preprocess, l2_normalize
from src.features.frequency_features import extract_frequency_feature
from src.models.clip_classifier import ClipClassifier
from src.models.frequency_classifier import FrequencyClassifier
from src.models.fusion_classifier import FusionClassifier
from src.models.resnet50_baseline import ResNet50Baseline


MODEL_ALIASES = {
    "resnet50": "resnet50_baseline",
    "resnet50_baseline": "resnet50_baseline",
    "clip": "clip_only",
    "clip_only": "clip_only",
    "frequency": "frequency_only",
    "frequency_only": "frequency_only",
    "fusion": "fusion",
}

PATH_COLUMNS = ["path", "filepath", "file_path", "image_path", "rel_path"]
LABEL_COLUMNS = ["label", "target", "class"]
CSV_COLUMNS = [
    "model",
    "corruption",
    "severity",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "num_samples",
    "num_positive",
    "num_negative",
    "status",
    "error",
    "checkpoint_path",
    "scaler_status",
    "missing_path_count",
    "elapsed_sec",
]

CHECKPOINT_CANDIDATES = {
    "fusion": [
        "artifacts/checkpoints/fusion.pt",
        "artifacts/checkpoints/fusion/fusion.pt",
        "outputs/checkpoints/fusion.pt",
        "outputs/checkpoints/fusion/best.pt",
        "outputs/checkpoints/fusion/best_checkpoint.pt",
    ],
    "frequency_only": [
        "artifacts/checkpoints/frequency_only.pt",
        "artifacts/checkpoints/frequency.pt",
        "outputs/checkpoints/frequency_only.pt",
        "outputs/checkpoints/frequency_only/best.pt",
    ],
    "clip_only": [
        "artifacts/checkpoints/clip_only.pt",
        "artifacts/checkpoints/clip.pt",
        "outputs/checkpoints/clip_only.pt",
        "outputs/checkpoints/clip_only/best.pt",
    ],
    "resnet50_baseline": [
        "artifacts/checkpoints/resnet50.pt",
        "artifacts/checkpoints/resnet50_baseline.pt",
        "outputs/checkpoints/resnet50.pt",
        "outputs/checkpoints/resnet50_baseline/best.pt",
        "outputs/genimage_tiny_full_resnet50_baseline/best_checkpoint.pt",
        "outputs/genimage_tiny_full_resnet50_baseline_28000/best_checkpoint.pt",
        "outputs/genimage_tiny_full_resnet50_baseline_28000train/best_checkpoint.pt",
    ],
}

T = TypeVar("T")


SCALER_CANDIDATES = [
    "artifacts/scalers/frequency_scaler.pkl",
    "artifacts/scalers/freq_scaler.pkl",
    "outputs/features/frequency_scaler.pkl",
    "outputs/checkpoints/frequency_scaler.pkl",
    "outputs/checkpoints/fusion_frequency_scaler.pkl",
]


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    row: dict[str, str]


@dataclass(frozen=True)
class ManifestData:
    path: Path
    samples: list[Sample]
    missing_path_count: int


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    corruption: str
    severity: str


class ImageBatchDataset(Dataset[tuple[Any, int, dict[str, str]]]):
    def __init__(self, images: Sequence[Image.Image], labels: Sequence[int], rows: Sequence[Mapping[str, str]], transform: Any) -> None:
        self.images = list(images)
        self.labels = [int(label) for label in labels]
        self.rows = [dict(row) for row in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]]:
        return self.transform(self.images[index]), self.labels[index], self.rows[index]


def _progress(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None, leave: bool = True) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        compressed.load()
        return compressed.convert("RGB")


def apply_resize_down_up(image: Image.Image, scale: float) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    down_size = (max(1, int(round(width * float(scale)))), max(1, int(round(height * float(scale)))))
    down = rgb.resize(down_size, resample=Image.Resampling.BICUBIC)
    return down.resize((width, height), resample=Image.Resampling.BICUBIC).convert("RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma))).convert("RGB")


def apply_center_crop_resize(image: Image.Image, crop_ratio: float = 0.85) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    crop_w = max(1, int(round(width * float(crop_ratio))))
    crop_h = max(1, int(round(height * float(crop_ratio))))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    cropped = rgb.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), resample=Image.Resampling.BICUBIC).convert("RGB")


def apply_corruption(image: Image.Image, spec: CorruptionSpec | str) -> Image.Image:
    if isinstance(spec, str):
        spec = _corruption_from_name(spec)
    if spec.corruption == "clean":
        return image.convert("RGB").copy()
    if spec.corruption == "jpeg":
        return apply_jpeg(image, int(spec.severity))
    if spec.corruption == "resize":
        return apply_resize_down_up(image, float(spec.severity))
    if spec.corruption == "blur":
        return apply_gaussian_blur(image, float(spec.severity))
    if spec.corruption == "center_crop_resize":
        return apply_center_crop_resize(image, float(spec.severity))
    raise ValueError(f"unsupported corruption: {spec.name}")


def resolve_checkpoint_path(model_name: str, config: dict[str, Any], checkpoint_dir: Path) -> Path:
    canonical = canonical_model_name(model_name)
    candidates = _checkpoint_candidates(canonical, config, checkpoint_dir)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"No checkpoint found for {canonical}. Tried:\n{tried}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real robustness evaluation for GenImage detector models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_csv", default="outputs/metrics/robustness_metrics.csv")
    parser.add_argument("--output_json", default="outputs/metrics/robustness_metrics.json")
    parser.add_argument("--plot_path", default="outputs/plots/robustness_barplot.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint_dir", default="artifacts/checkpoints")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_config(config_path)
    print(f"Loaded config: {config_path}")
    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    models = parse_models(args.models)
    print(f"Models requested: {', '.join(models)}")

    data_root = Path(args.data_root) if args.data_root else _default_data_root(config)
    manifest = load_manifest(config, cli_manifest=args.manifest, data_root=data_root, max_samples=args.max_samples)
    print(f"Resolved manifest: {manifest.path}")
    print(f"Valid test images: {len(manifest.samples)}")
    print(f"Missing image paths: {manifest.missing_path_count}")

    resolved: dict[str, Path] = {}
    for model in models:
        try:
            checkpoint = resolve_checkpoint_path(model, config, Path(args.checkpoint_dir))
            resolved[model] = checkpoint
            print(f"Resolved checkpoint for {model}: {checkpoint}")
        except FileNotFoundError as exc:
            print(str(exc))

    if args.dry_run:
        return 0 if resolved else 1

    rows: list[dict[str, Any]] = []
    for model in _progress(models, desc="Robustness models", unit="model"):
        checkpoint = resolved.get(model)
        if checkpoint is None:
            error = _missing_checkpoint_error(model, config, Path(args.checkpoint_dir))
            rows.extend(_failed_rows(model, error, manifest.missing_path_count, "", "not_applicable"))
            continue
        try:
            runner = ModelRunner(model, checkpoint, config, device)
        except Exception as exc:
            rows.extend(_failed_rows(model, str(exc), manifest.missing_path_count, checkpoint.as_posix(), "not_loaded"))
            continue
        for spec in _progress(corruption_specs(), desc=f"{model} corruptions", unit="corruption"):
            start = time.perf_counter()
            print(f"Running model={model} corruption={spec.name} samples={len(manifest.samples)}")
            try:
                images = [
                    _load_and_corrupt(sample.path, spec)
                    for sample in _progress(manifest.samples, desc=f"{model} {spec.name} images", unit="image", leave=False)
                ]
                labels = np.asarray([sample.label for sample in manifest.samples], dtype=np.int64)
                probs = runner.predict(images, [sample.row for sample in manifest.samples], labels)
                metrics = compute_metrics(labels, probs, threshold=_threshold(config))
                rows.append({
                    "model": model,
                    "corruption": spec.corruption,
                    "severity": spec.severity,
                    **metrics,
                    "status": "ok",
                    "error": "",
                    "checkpoint_path": checkpoint.as_posix(),
                    "scaler_status": runner.scaler_status,
                    "missing_path_count": manifest.missing_path_count,
                    "elapsed_sec": round(time.perf_counter() - start, 3),
                })
            except Exception as exc:
                rows.append(_failed_row(model, spec, str(exc), manifest.missing_path_count, checkpoint.as_posix(), runner.scaler_status, time.perf_counter() - start))

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_outputs(rows, output_csv, output_json)
    print(f"Saved robustness metrics to {output_csv}")
    try:
        write_plot(rows, Path(args.plot_path))
        print(f"Saved robustness plot to {args.plot_path}")
    except Exception as exc:
        print(f"Plot generation failed: {exc}")
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


class ModelRunner:
    def __init__(self, model_name: str, checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.device = device
        self.checkpoint = _torch_load(checkpoint_path)
        self.scaler = load_frequency_scaler(config, self.checkpoint)
        self.scaler_status = "loaded" if self.scaler is not None else "missing_raw_used"
        self.model = self._build_model().to(device)
        self.model.eval()
        self.clip_model: Any | None = None
        self.clip_preprocess: Any | None = None
        if model_name in {"clip_only", "fusion"}:
            self.clip_model, self.clip_preprocess = load_clip_model_and_preprocess(config, device)

    def predict(self, images: Sequence[Image.Image], rows: Sequence[Mapping[str, str]], labels: np.ndarray) -> np.ndarray:
        if self.model_name == "frequency_only":
            features = self._frequency_features(images)
        elif self.model_name == "clip_only":
            features = self._clip_features(images, rows, labels)
        elif self.model_name == "fusion":
            clip_features = self._clip_features(images, rows, labels)
            freq_features = self._frequency_features(images)
            features = np.concatenate([clip_features, freq_features], axis=1).astype(np.float32, copy=False)
        elif self.model_name == "resnet50_baseline":
            return self._resnet_probs(images, labels, rows)
        else:
            raise ValueError(f"unsupported model: {self.model_name}")
        return self._feature_probs(features)

    def _build_model(self) -> nn.Module:
        state = _state_dict(self.checkpoint)
        input_dim = _checkpoint_int(self.checkpoint, "input_dim", state)
        hidden_dim = int(self.checkpoint.get("hidden_dim", self.config.get("classifier", {}).get("hidden_dim", 512)))
        dropout = float(self.config.get("classifier", {}).get("dropout", 0.2))
        if self.model_name == "frequency_only":
            model: nn.Module = FrequencyClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif self.model_name == "clip_only":
            model = ClipClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif self.model_name == "fusion":
            clip_dim = int(self.config.get("clip", {}).get("output_dim", 768))
            freq_dim = max(1, input_dim - clip_dim)
            model = FusionClassifier(clip_dim=clip_dim, freq_dim=freq_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif self.model_name == "resnet50_baseline":
            model = ResNet50Baseline(pretrained=False)
        else:
            raise ValueError(f"unsupported model: {self.model_name}")
        _load_state(model, state)
        return model

    def _frequency_features(self, images: Sequence[Image.Image]) -> np.ndarray:
        features = np.stack(
            [extract_frequency_feature(image, self.config) for image in _progress(images, desc=f"{self.model_name} frequency features", unit="image", leave=False)],
            axis=0,
        ).astype(np.float32, copy=False)
        if self.scaler is not None:
            transformed = self.scaler.transform(features)
            return np.asarray(transformed, dtype=np.float32)
        print("Warning: frequency scaler missing; using raw frequency features")
        return features

    def _clip_features(self, images: Sequence[Image.Image], rows: Sequence[Mapping[str, str]], labels: np.ndarray) -> np.ndarray:
        if self.clip_model is None or self.clip_preprocess is None:
            raise ClipModelLoadError("CLIP model was not loaded")
        dataset = ImageBatchDataset(images, labels.tolist(), rows, self.clip_preprocess)
        loader = DataLoader(dataset, batch_size=int(self.config.get("data", {}).get("batch_size", 16)), shuffle=False)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for image_batch, _label_batch, _meta in _progress(loader, desc=f"{self.model_name} CLIP batches", total=len(loader), unit="batch", leave=False):
                encoded = self.clip_model.encode_image(image_batch.to(self.device))
                batches.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
        if not batches:
            raise ValueError("CLIP feature extraction produced no batches")
        features = np.concatenate(batches, axis=0)
        if bool(self.config.get("clip", {}).get("normalize_feature", True)):
            features = l2_normalize(features)
        return features.astype(np.float32, copy=False)

    def _feature_probs(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(self.device)
        with torch.inference_mode():
            probs = torch.sigmoid(self.model(tensor)).detach().cpu().numpy()
        return np.asarray(probs, dtype=np.float64).reshape(-1)

    def _resnet_probs(self, images: Sequence[Image.Image], labels: np.ndarray, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
        transform = get_eval_transform(int(self.config.get("data", {}).get("image_size", 512)))
        dataset = ImageBatchDataset(images, labels.tolist(), rows, transform)
        loader = DataLoader(dataset, batch_size=int(self.config.get("data", {}).get("batch_size", 16)), shuffle=False)
        probs: list[np.ndarray] = []
        with torch.inference_mode():
            for image_batch, _label_batch, _meta in _progress(loader, desc=f"{self.model_name} ResNet batches", total=len(loader), unit="batch", leave=False):
                logits = self.model(image_batch.to(self.device))
                probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(probs).astype(np.float64, copy=False).reshape(-1)


def load_manifest(config: dict[str, Any], *, cli_manifest: str | None, data_root: Path, max_samples: int | None) -> ManifestData:
    manifest_path = resolve_manifest_path(config, cli_manifest)
    rows = _read_csv_rows(manifest_path)
    path_col = _first_present(rows, PATH_COLUMNS)
    label_col = _first_present(rows, LABEL_COLUMNS)
    if path_col is None or label_col is None:
        raise SystemExit(f"Manifest {manifest_path} must contain path and label columns")
    rows = _select_split_rows(rows)
    samples: list[Sample] = []
    missing = 0
    basename_index: dict[str, list[Path]] | None = None
    for row in _progress(rows, desc="Resolving manifest images", unit="row"):
        resolved = _resolve_image_path(row[path_col], data_root, basename_index, row.get("root"))
        if resolved is None:
            if basename_index is None:
                basename_index = _basename_index(data_root)
                resolved = _resolve_image_path(row[path_col], data_root, basename_index, row.get("root"))
        if resolved is None:
            missing += 1
            continue
        try:
            label = _label_value(row[label_col])
        except ValueError:
            missing += 1
            continue
        normalized = dict(row)
        normalized["filepath"] = resolved.as_posix()
        normalized["label"] = str(label)
        samples.append(Sample(resolved, label, normalized))
    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        samples = _limit_samples(samples, max_samples)
    if not samples:
        raise SystemExit("No valid test images found. Check --manifest and --data_root.")
    return ManifestData(manifest_path, samples, missing)


def _limit_samples(samples: list[Sample], max_samples: int) -> list[Sample]:
    by_label = {0: [sample for sample in samples if sample.label == 0], 1: [sample for sample in samples if sample.label == 1]}
    if by_label[0] and by_label[1]:
        first_count = max_samples // 2
        selected = by_label[0][:first_count] + by_label[1][: max_samples - first_count]
        if len(selected) < max_samples:
            selected_ids = {id(sample) for sample in selected}
            selected.extend(sample for sample in samples if id(sample) not in selected_ids)
        return selected[:max_samples]
    return samples[:max_samples]


def resolve_manifest_path(config: dict[str, Any], cli_manifest: str | None) -> Path:
    candidates: list[Path] = []
    if cli_manifest:
        candidates.append(Path(cli_manifest))
    candidates.extend(_config_manifest_candidates(config))
    candidates.extend([
        Path("data/genimage_manifest.csv"),
        Path("data/tiny_genimage_manifest.csv"),
        Path("artifacts/metadata/test.csv"),
        Path("artifacts/metadata/manifest.csv"),
        Path("outputs/genimage_tiny_full/manifest.csv"),
        Path("data/metadata/genimage_tiny_full_dataset.csv"),
        Path("data/metadata/genimage_tiny_dataset.csv"),
    ])
    for candidate in candidates:
        path = _project_path(candidate)
        if path.is_file():
            return path
    raise SystemExit("No manifest found. Tried:\n" + "\n".join(f"- {_project_path(path)}" for path in candidates))


def canonical_model_name(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"unsupported model {name!r}; expected one of {sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[key]


def parse_models(value: str) -> list[str]:
    models = [canonical_model_name(item) for item in value.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must include at least one model")
    return list(dict.fromkeys(models))


def corruption_specs() -> list[CorruptionSpec]:
    return [
        CorruptionSpec("clean", "clean", "none"),
        CorruptionSpec("jpeg_q95", "jpeg", "95"),
        CorruptionSpec("jpeg_q75", "jpeg", "75"),
        CorruptionSpec("jpeg_q50", "jpeg", "50"),
        CorruptionSpec("resize_0.5", "resize", "0.5"),
        CorruptionSpec("resize_0.25", "resize", "0.25"),
        CorruptionSpec("blur_1.0", "blur", "1.0"),
        CorruptionSpec("blur_2.0", "blur", "2.0"),
        CorruptionSpec("center_crop_resize", "center_crop_resize", "0.85"),
    ]


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(np.int64)
    precision, recall, f1, _support = precision_recall_fscore_support(labels, preds, average="binary", zero_division="warn")
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": math.nan,
        "average_precision": math.nan,
        "num_samples": int(labels.size),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
    }
    if out["num_positive"] and out["num_negative"]:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["average_precision"] = float(average_precision_score(labels, probs))
    return out


def load_frequency_scaler(config: dict[str, Any], checkpoint: Mapping[str, Any] | None = None) -> Any | None:
    candidates = _checkpoint_scaler_candidates(checkpoint)
    if not candidates and checkpoint is None:
        candidates = _config_scaler_candidates(config) + [Path(path) for path in SCALER_CANDIDATES]
    for candidate in candidates:
        path = _project_path(candidate)
        if path.is_file():
            return joblib.load(path)
    return None


def _checkpoint_scaler_candidates(checkpoint: Mapping[str, Any] | None) -> list[Path]:
    if checkpoint is None:
        return []
    snapshot = checkpoint.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        snapshot = checkpoint.get("config")
    if not isinstance(snapshot, Mapping):
        return []
    candidates: list[Path] = []
    for section_name in ("paths", "frequency", "features"):
        section = snapshot.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key in ("frequency_scaler_path", "scaler_path", "scaler", "frequency_scaler"):
            value = section.get(key)
            if value and str(value) != "standard":
                candidates.append(Path(str(value)))
    return candidates


def write_outputs(rows: Sequence[Mapping[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows([{key: _csv_value(row.get(key, "")) for key in CSV_COLUMNS} for row in rows])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file_obj:
        json.dump([{key: _json_value(row.get(key, "")) for key in CSV_COLUMNS} for row in rows], file_obj, indent=2)
        file_obj.write("\n")


def write_plot(rows: Sequence[Mapping[str, Any]], plot_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if not ok_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no successful robustness rows", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return
    frame = pd.DataFrame(ok_rows)
    frame["plot_label"] = frame.apply(lambda row: str(row["corruption"]) if str(row["severity"]) == "none" else f"{row['corruption']}_{row['severity']}", axis=1)
    pivot = frame.pivot(index="plot_label", columns="model", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("accuracy")
    ax.set_xlabel("corruption")
    ax.set_ylim(0.0, 1.0)
    ax.figure.tight_layout()
    ax.figure.savefig(plot_path)
    plt.close(ax.figure)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return loaded


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _default_data_root(config: dict[str, Any]) -> Path:
    data = config.get("data", {})
    paths = config.get("paths", {})
    for value in [data.get("data_root") if isinstance(data, dict) else None, paths.get("data_root") if isinstance(paths, dict) else None]:
        if value:
            return Path(str(value))
    return Path("data")


def _config_manifest_candidates(config: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for section_name in ("paths", "data", "eval"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("manifest", "manifest_csv", "manifest_path", "test_csv", "dataset_csv"):
            value = section.get(key)
            if value:
                candidates.append(Path(str(value)))
    return candidates


def _checkpoint_candidates(model_name: str, config: dict[str, Any], checkpoint_dir: Path) -> list[Path]:
    raw: list[Path] = []
    raw.extend(_config_checkpoint_candidates(model_name, config))
    raw.append(checkpoint_dir / f"{model_name}.pt")
    if model_name == "fusion":
        raw.append(checkpoint_dir / "fusion.pt")
    raw.extend(Path(path) for path in CHECKPOINT_CANDIDATES[model_name])
    candidates: list[Path] = []
    seen: set[str] = set()
    for item in raw:
        path = _project_path(item)
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    return candidates


def _config_checkpoint_candidates(model_name: str, config: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    paths = config.get("paths", {})
    if isinstance(paths, dict):
        checkpoint_dir = paths.get("checkpoint_dir")
        if checkpoint_dir:
            base = Path(str(checkpoint_dir))
            candidates.append(base / f"{model_name}.pt")
            candidates.append(base / "best_checkpoint.pt")
            if model_name == "fusion":
                candidates.append(base / "fusion.pt")
        for key in (f"{model_name}_checkpoint", "checkpoint", "checkpoint_path"):
            value = paths.get(key)
            if value:
                candidates.append(Path(str(value)))
    models = config.get("models", {})
    if isinstance(models, dict) and isinstance(models.get(model_name), dict):
        value = models[model_name].get("checkpoint") or models[model_name].get("checkpoint_path")
        if value:
            candidates.append(Path(str(value)))
    return candidates


def _config_scaler_candidates(config: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for section_name in ("paths", "frequency", "features"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("scaler", "scaler_path", "frequency_scaler", "frequency_scaler_path"):
            value = section.get(key)
            if value:
                candidates.append(Path(str(value)))
    return candidates


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return [dict(row) for row in csv.DictReader(file_obj)]


def _first_present(rows: Sequence[Mapping[str, str]], names: Sequence[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for name in names:
        if name in keys:
            return name
    return None


def _select_split_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows or "split" not in rows[0]:
        return rows
    test_rows = [row for row in rows if row.get("split") == "test"]
    if test_rows:
        return test_rows
    val_rows = [row for row in rows if row.get("split") == "val"]
    return val_rows or rows


def _resolve_image_path(value: str, data_root: Path, basename_index: dict[str, list[Path]] | None, root_value: str | None = None) -> Path | None:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [_project_path(raw), _project_path(data_root / raw)]
    if root_value and not raw.is_absolute():
        candidates.insert(0, _project_path(Path(root_value) / raw))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if basename_index is not None:
        matches = basename_index.get(raw.name, [])
        if matches:
            return matches[0]
    return None


def _basename_index(data_root: Path) -> dict[str, list[Path]]:
    root = _project_path(data_root)
    if not root.exists():
        return {}
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def _label_value(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"1", "fake", "ai", "generated", "synthetic"}:
        return 1
    if text in {"0", "real", "original", "human", "natural"}:
        return 0
    number = int(float(text))
    if number not in {0, 1}:
        raise ValueError(f"unsupported label: {value}")
    return number


def _load_and_corrupt(path: Path, spec: CorruptionSpec) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return apply_corruption(image, spec)


def _torch_load(path: Path) -> dict[str, Any]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {path}")
    return loaded


def _state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint missing model_state_dict")
    return state


def _checkpoint_int(checkpoint: Mapping[str, Any], key: str, state: Mapping[str, torch.Tensor]) -> int:
    value = checkpoint.get(key)
    if value is not None:
        return int(value)
    for weight_key, tensor in state.items():
        if weight_key.endswith("0.weight") or weight_key.endswith("input.0.weight") or weight_key.endswith("net.0.weight"):
            return int(tensor.shape[1])
    raise ValueError(f"checkpoint missing {key}")


def _load_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as first_error:
        prefixed = {f"classifier.{key}": value for key, value in state.items()}
        try:
            model.load_state_dict(prefixed, strict=True)
        except RuntimeError:
            raise first_error


def _threshold(config: dict[str, Any]) -> float:
    evaluation = config.get("eval")
    if isinstance(evaluation, dict):
        return float(evaluation.get("threshold", 0.5))
    return 0.5


def _failed_rows(model: str, error: str, missing_path_count: int, checkpoint_path: str, scaler_status: str) -> list[dict[str, Any]]:
    return [_failed_row(model, spec, error, missing_path_count, checkpoint_path, scaler_status, 0.0) for spec in corruption_specs()]


def _failed_row(model: str, spec: CorruptionSpec, error: str, missing_path_count: int, checkpoint_path: str, scaler_status: str, elapsed: float) -> dict[str, Any]:
    return {
        "model": model,
        "corruption": spec.corruption,
        "severity": spec.severity,
        "accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "f1": math.nan,
        "roc_auc": math.nan,
        "average_precision": math.nan,
        "num_samples": 0,
        "num_positive": 0,
        "num_negative": 0,
        "status": "failed",
        "error": error,
        "checkpoint_path": checkpoint_path,
        "scaler_status": scaler_status,
        "missing_path_count": missing_path_count,
        "elapsed_sec": round(elapsed, 3),
    }


def _missing_checkpoint_error(model: str, config: dict[str, Any], checkpoint_dir: Path) -> str:
    try:
        resolve_checkpoint_path(model, config, checkpoint_dir)
    except FileNotFoundError as exc:
        return str(exc)
    return "checkpoint not resolved"


def _corruption_from_name(name: str) -> CorruptionSpec:
    for spec in corruption_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"unsupported corruption: {name}")


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
