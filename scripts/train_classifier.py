from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

import argparse
import sys
from pathlib import Path

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.features.cache import FeatureCacheError, TorchDependencyError  # noqa: E402
from src.train.frequency_lr import train_classifier, verify_reload_equivalence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier artifacts from assembled frequency, CLIP, or fusion features.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest v1 CSV matching the feature caches.")
    parser.add_argument("--feature_cache", type=Path, default=None, help="Backward-compatible alias for --frequency_cache.")
    parser.add_argument("--frequency_cache", type=Path, default=None, help="Validated feature_type=frequency cache .pt file.")
    parser.add_argument("--clip_cache", type=Path, default=None, help="Validated feature_type=clip cache .pt file.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Experiment artifact directory to create or update.")
    parser.add_argument(
        "--mode",
        choices=["frequency_only", "clip_only", "fusion"],
        default="frequency_only",
        help="Feature mode assembled with train-only frequency scaling where applicable.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "linear_svm"],
        default="logistic_regression",
        help="Classifier to train. Linear SVM artifacts are decision-score-only unless calibrated probability is added later.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sklearn random_state and config seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for LogisticRegression pred_label; not selected on test.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Classifier max_iter.")
    parser.add_argument("--C", dest="c_value", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument("--verify_reload", action=argparse.BooleanOptionalAction, default=True, help="Reload artifacts and verify predictions match.")
    parser.add_argument("--reload_tolerance", type=float, default=1e-12, help="Max absolute diff allowed for reload equivalence.")
    parser.add_argument("--verify_only", action="store_true", help="Only verify reload equivalence for an existing output_dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frequency_cache = args.frequency_cache or args.feature_cache
    try:
        if args.verify_only:
            max_abs_diff = verify_reload_equivalence(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                mode=args.mode,
                frequency_cache_path=frequency_cache,
                clip_cache_path=args.clip_cache,
                tolerance=args.reload_tolerance,
            )
            print(f"reload equivalence passed: max_abs_diff={max_abs_diff:.6g}")
            return 0

        result = train_classifier(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            mode=args.mode,
            classifier=args.classifier,
            frequency_cache_path=frequency_cache,
            clip_cache_path=args.clip_cache,
            seed=args.seed,
            threshold=args.threshold,
            max_iter=args.max_iter,
            c_value=args.c_value,
            verify_reload=args.verify_reload,
            reload_tolerance=args.reload_tolerance,
            command=sys.argv,
        )
        print(f"wrote model: {result.model_path}")
        print(f"wrote scaler: {result.scaler_path}")
        print(f"wrote config: {result.config_path}")
        print(f"wrote metrics: {result.metrics_path}")
        print(f"wrote predictions: {result.predictions_path}")
        if result.reload_max_abs_diff is not None:
            print(f"reload equivalence passed: max_abs_diff={result.reload_max_abs_diff:.6g}")
        return 0
    except (FeatureCacheError, TorchDependencyError, ValueError, RuntimeError) as error:
        print(f"classifier training failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
