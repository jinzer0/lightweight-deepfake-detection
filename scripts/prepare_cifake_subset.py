from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/prepare_cifake_subset.py creates legacy manifest-v1 CSV files.

Use the current canonical dataset.csv workflow instead. For the CPU-safe smoke dataset:
  python -m src.data.make_dummy_dataset --num_real 30 --num_fake 30 --output_dir data/raw/dummy --csv data/metadata/dataset.csv
  python -m src.data.validate_metadata --csv data/metadata/dataset.csv

For real local data, create data/metadata/dataset.csv with the README schema, then validate it with src.data.validate_metadata.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
