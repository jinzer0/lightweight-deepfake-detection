from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/extract_clip_features.py is a legacy manifest-v1/.pt cache entrypoint.

Use the current dataset.csv + .npy cache workflow instead. Frequency is the mandatory CPU-safe path:
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test

Optional CLIP cache commands, only when open_clip_torch and model weights are available:
  python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split train
  python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split val
  python -m src.features.cache_features --config configs/default.yaml --feature_type clip --split test

See README.md before running optional CLIP paths offline.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
