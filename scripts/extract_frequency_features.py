from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/extract_frequency_features.py is a legacy manifest-v1/.pt cache entrypoint.

Use the current dataset.csv + .npy feature cache workflow instead:
  python -m src.data.validate_metadata --csv data/metadata/dataset.csv
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split train
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split val
  python -m src.features.cache_features --config configs/default.yaml --feature_type frequency --split test

See README.md for the full CPU-safe frequency smoke sequence.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
