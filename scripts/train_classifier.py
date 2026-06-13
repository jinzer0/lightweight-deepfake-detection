from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/train_classifier.py is a legacy classifier entrypoint.

Use the current dataset.csv + .npy + PyTorch workflow instead:
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test
  python -m src.train.train_frequency --config configs/default.yaml

See README.md for the full CPU-safe smoke sequence.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
