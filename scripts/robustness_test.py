from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

CORRUPTIONS = [
    "jpeg_q95", "jpeg_q75", "jpeg_q50", "resize_0.5", "resize_0.25",
    "gaussian_blur_1.0", "gaussian_blur_2.0", "center_crop_resize",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run robustness evaluation for GenImage detector models.")
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument("--models", default="resnet50,clip_only,frequency_only,fusion")
    p.add_argument("--output", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    metrics_dir = Path(cfg.get("outputs", {}).get("metrics_dir", "outputs/metrics"))
    plots_dir = Path(cfg.get("outputs", {}).get("plots_dir", "outputs/plots"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        canonical = "resnet50_baseline" if model == "resnet50" else model
        for corruption in CORRUPTIONS:
            rows.append({"model": canonical, "corruption": corruption, "status": "not run", "accuracy": "not run", "roc_auc": "not run"})
    out = Path(args.output) if args.output else metrics_dir / "robustness_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "not run", ha="center")
    ax.set_axis_off()
    fig.savefig(plots_dir / "robustness_barplot.png")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
