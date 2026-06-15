from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.metrics import average_precision_score_binary, compute_binary_metrics
from src.features.cache import FeatureCacheError, load_split_feature_bundle, load_split_features, assert_aligned_feature_rows
from src.models.clip_classifier import ClipClassifier
from src.models.frequency_classifier import FrequencyClassifier
from src.models.fusion_mlp import FusionMLP

MODELS = ["resnet50_baseline", "clip_only", "frequency_only", "fusion"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate GenImage detector artifacts and compare models.")
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument("--split", default="test")
    return p.parse_args(argv)


def not_run_row(model: str, split: str, reason: str) -> dict[str, Any]:
    return {"model": model, "split": split, "status": "not run", "reason": reason, "accuracy": "not run", "precision": "not run", "recall": "not run", "f1": "not run", "roc_auc": "not run", "average_precision": "not run", "inference_time_per_image": "not run"}


def main(argv=None):
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    paths = cfg.get("paths", {})
    metrics_dir = Path(paths.get("report_dir", cfg.get("outputs", {}).get("metrics_dir", "outputs/metrics")))
    if metrics_dir.name == "reports":
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_metrics_dir = Path("outputs/metrics"); output_metrics_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_metrics_dir = metrics_dir; output_metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(cfg.get("outputs", {}).get("plots_dir", "outputs/plots")); plots_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(paths.get("feature_dir", cfg.get("outputs", {}).get("features_dir", "artifacts/features")))
    checkpoint_root = Path(paths.get("checkpoint_root", "artifacts/checkpoints"))
    rows: list[dict[str, Any]] = []
    per_generator_rows: list[dict[str, Any]] = []
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for model_name in MODELS:
        try:
            if model_name == "resnet50_baseline":
                rows.append(not_run_row(model_name, args.split, "image-level ResNet evaluation requires manifest image inference and is not run by this lightweight evaluator"))
                continue
            y_true, y_prob, generators = _predict_feature_model(model_name, feature_dir, checkpoint_root, args.split)
            metrics = compute_binary_metrics(y_true, y_prob)
            row = {"model": model_name, "split": args.split, "status": "ok", "reason": "", **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
            row["average_precision"] = average_precision_score_binary(y_true, y_prob)
            row["inference_time_per_image"] = "not measured"
            rows.append(row)
            _save_confusion(plots_dir / f"confusion_matrix_{model_name.replace('_baseline','')}.png", metrics["confusion_matrix"], model_name)
            curves.append((model_name, y_true, y_prob))
            for generator in sorted(set(generators)):
                mask = np.asarray(generators) == generator
                gm = compute_binary_metrics(y_true[mask], y_prob[mask])
                per_generator_rows.append({"model": model_name, "generator": generator, "accuracy": gm["accuracy"], "roc_auc": gm["roc_auc"]})
        except (FileNotFoundError, FeatureCacheError, RuntimeError, ValueError, KeyError) as exc:
            rows.append(not_run_row(model_name, args.split, str(exc)))
    for model_name in MODELS:
        path = plots_dir / f"confusion_matrix_{model_name.replace('_baseline','')}.png"
        if not path.exists():
            _save_placeholder(path, "not run")
    pd.DataFrame(rows).to_csv(output_metrics_dir / "model_comparison.csv", index=False)
    (output_metrics_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(per_generator_rows or [{"model": "not run", "generator": "not run", "accuracy": "not run", "roc_auc": "not run"}]).to_csv(output_metrics_dir / "per_generator_metrics.csv", index=False)
    _save_curve_placeholder(plots_dir / "roc_curve_all_models.png", curves, "ROC")
    _save_curve_placeholder(plots_dir / "pr_curve_all_models.png", curves, "PR")
    print(f"wrote {output_metrics_dir / 'model_comparison.csv'}")
    return 0


def _predict_feature_model(model_name: str, feature_dir: Path, checkpoint_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if model_name == "clip_only":
        x, y, rows = load_split_feature_bundle(feature_dir, "clip", split)
        model = ClipClassifier(input_dim=x.shape[1])
    elif model_name == "frequency_only":
        x, y, rows = load_split_feature_bundle(feature_dir, "frequency", split)
        model = FrequencyClassifier(input_dim=x.shape[1])
    else:
        clip_x, y, clip_rows = load_split_feature_bundle(feature_dir, "clip", split)
        freq_x, freq_y, freq_rows = load_split_feature_bundle(feature_dir, "frequency", split)
        assert_aligned_feature_rows(clip_rows, freq_rows)
        if not np.array_equal(y, freq_y):
            raise ValueError("clip and frequency labels are not aligned")
        x = np.concatenate([clip_x, freq_x], axis=1).astype(np.float32)
        rows = clip_rows
        model = FusionMLP(clip_dim=clip_x.shape[1], freq_dim=freq_x.shape[1])
    ckpt = checkpoint_root / model_name / "best_checkpoint.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        probs = torch.sigmoid(model(torch.from_numpy(x).float())).numpy()
    generators = [row.get("generator", "unknown") for row in rows]
    return y.astype(np.int64), probs.astype(np.float64), generators


def _save_confusion(path: Path, matrix: Any, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.asarray(matrix, dtype=float), cmap="Blues")
    ax.set_title(title); ax.set_xlabel("pred"); ax.set_ylabel("true")
    fig.savefig(path); plt.close(fig)


def _save_placeholder(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 3)); ax.text(0.5, 0.5, text, ha="center"); ax.set_axis_off(); fig.savefig(path); plt.close(fig)


def _save_curve_placeholder(path: Path, curves: list[tuple[str, np.ndarray, np.ndarray]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    if not curves:
        ax.text(0.5, 0.5, "not run", ha="center"); ax.set_axis_off()
    else:
        for name, _y, prob in curves:
            ax.plot(np.sort(prob), label=name)
        ax.legend(); ax.set_title(title)
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
