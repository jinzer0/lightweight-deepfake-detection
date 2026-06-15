from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.evaluate import MODEL_SPECS, EvaluationResult, evaluate_model
from src.eval.metrics import average_precision_score_binary
from src.features.cache_features import NpyFeatureCacheError
from src.models.checkpoint import CheckpointError
from src.utils.config import load_config

MODELS = ["resnet50_baseline", "clip_only", "frequency_only", "fusion"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate GenImage detector artifacts and compare models.")
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument("--split", default="test")
    return p.parse_args(argv)


def not_run_row(model: str, split: str, reason: str) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "status": "not run",
        "reason": reason,
        "accuracy": "not run",
        "precision": "not run",
        "recall": "not run",
        "f1": "not run",
        "roc_auc": "not run",
        "average_precision": "not run",
        "inference_time_per_image": "not run",
    }


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    output_metrics_dir = Path(str(paths.get("report_dir", "outputs/metrics")))
    output_metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(str(cfg.get("outputs", {}).get("plots_dir", "outputs/plots"))) if isinstance(cfg.get("outputs"), dict) else Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_generator_rows: list[dict[str, Any]] = []
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for model_name in MODELS:
        try:
            if model_name == "resnet50_baseline":
                rows.append(not_run_row(model_name, args.split, "image-level ResNet evaluation requires manifest image inference and is not run by this lightweight evaluator"))
                continue
            result = evaluate_model(cfg, model_name=model_name, split=args.split)
            labels, probabilities, generators = _prediction_data(result)
            metrics = dict(result.metrics["metrics"])
            row = {"model": model_name, "split": args.split, "status": "ok", "reason": ""}
            row.update({key: metrics.get(key) for key in ("accuracy", "precision", "recall", "f1", "roc_auc")})
            row["average_precision"] = average_precision_score_binary(labels, probabilities)
            row["inference_time_per_image"] = "not measured"
            rows.append(row)
            _save_confusion(plots_dir / f"confusion_matrix_{model_name.replace('_baseline','')}.png", metrics.get("confusion_matrix"), model_name)
            curves.append((model_name, labels, probabilities))
            for generator in sorted(set(generators)):
                mask = np.asarray(generators) == generator
                generator_metrics = _generator_metrics(result, generator)
                per_generator_rows.append(
                    {
                        "model": model_name,
                        "generator": generator,
                        "accuracy": generator_metrics.get("accuracy", "not run") if generator_metrics else "not run",
                        "roc_auc": generator_metrics.get("roc_auc", "not run") if generator_metrics else "not run",
                        "count": int(np.sum(mask)),
                    }
                )
        except (CheckpointError, FileNotFoundError, NpyFeatureCacheError, RuntimeError, ValueError, KeyError) as exc:
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
    return 0 if any(row.get("status") == "ok" for row in rows) else 1


def _prediction_data(result: EvaluationResult) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(result.predictions_path)
    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = frame["pred_prob"].to_numpy(dtype=np.float64)
    generators = frame["generator"].astype(str).tolist() if "generator" in frame.columns else ["unknown"] * len(frame)
    return labels, probabilities, generators


def _generator_metrics(result: EvaluationResult, generator: str) -> dict[str, Any]:
    per_generator = result.metrics.get("per_generator", [])
    if not isinstance(per_generator, list):
        return {}
    for row in per_generator:
        if isinstance(row, dict) and str(row.get("generator")) == generator:
            return row
    return {}


def _save_confusion(path: Path, matrix: Any, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.asarray(matrix, dtype=float), cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    fig.savefig(path)
    plt.close(fig)


def _save_placeholder(path: Path, text: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.text(0.5, 0.5, text, ha="center")
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def _save_curve_placeholder(path: Path, curves: list[tuple[str, np.ndarray, np.ndarray]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    if not curves:
        ax.text(0.5, 0.5, "not run", ha="center")
        ax.set_axis_off()
    else:
        for name, _y, prob in curves:
            ax.plot(np.sort(prob), label=name)
        ax.legend()
        ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
