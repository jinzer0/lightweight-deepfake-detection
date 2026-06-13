from __future__ import annotations

import argparse

# pyright: reportAny=false, reportUnusedCallResult=false

from src.features.cache_features import NpyFeatureCacheError
from src.train.common import TrainerSettings, train_feature_mlp
from src.utils.config import load_config


def train_frequency(config: dict[str, object]) -> None:
    _ = train_feature_mlp(config, TrainerSettings(feature_type="frequency", artifact_stem="frequency_only"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP on cached frequency .npy features.")
    parser.add_argument("--config", required=True, help="Path to project YAML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    try:
        train_frequency(config)
    except NpyFeatureCacheError as exc:
        raise SystemExit(f"Frequency training failed clearly: {exc}") from None


if __name__ == "__main__":
    main()
