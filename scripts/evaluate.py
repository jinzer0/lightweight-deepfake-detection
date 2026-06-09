from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

import argparse
import sys
from pathlib import Path

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.eval import ArtifactValidationError, evaluate_experiment, validate_experiment_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh metrics and plots from an existing experiment predictions.csv.")
    parser.add_argument("--experiment_dir", type=Path, required=True, help="Existing experiment artifact directory.")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Optional split to evaluate.")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=False, help="Validate artifacts after refreshing.")
    parser.add_argument("--validate_only", action="store_true", help="Validate existing artifacts without refreshing metrics or plots.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.validate_only:
            validate_experiment_artifacts(args.experiment_dir)
            print(f"artifact validation passed: {args.experiment_dir}")
            return 0
        metrics = evaluate_experiment(args.experiment_dir, split=args.split, validate=args.validate)
        print(f"wrote metrics: {args.experiment_dir / 'metrics.json'}")
        print(f"wrote confusion matrix: {args.experiment_dir / 'confusion_matrix.png'}")
        print(f"wrote ROC curve: {args.experiment_dir / 'roc_curve.png'}")
        print(f"wrote PR curve: {args.experiment_dir / 'pr_curve.png'}")
        print(f"sample_count={metrics['sample_count']} ranking_metric_input={metrics['ranking_metric_input']}")
        return 0
    except (ArtifactValidationError, OSError, TypeError, ValueError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
