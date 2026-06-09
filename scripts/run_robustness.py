from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

import argparse
from pathlib import Path

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.eval.robustness import run_frequency_robustness  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JPEG, resize, and blur robustness evaluation for clean-trained artifacts.")
    _ = parser.add_argument("--manifest", type=Path, required=True, help="Manifest v1 CSV containing the clean test images.")
    _ = parser.add_argument("--experiment_dir", type=Path, required=True, help="Clean-trained experiment artifact directory.")
    _ = parser.add_argument("--output_dir", type=Path, default=None, help="Output directory for robustness_metrics.csv and robustness_summary.png.")
    _ = parser.add_argument("--quick", action="store_true", help="Use a tiny test subset and one level per corruption family.")
    _ = parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on selected test samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_frequency_robustness(
        manifest_path=args.manifest,
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        mode="quick" if args.quick else "full",
        max_samples=args.max_samples,
    )
    print(f"Wrote robustness metrics: {result.metrics_path}")
    print(f"Wrote robustness summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
