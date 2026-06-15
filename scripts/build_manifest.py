from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts._path import bootstrap
bootstrap()

from src.data.manifest import GENIMAGE_MANIFEST_COLUMNS, build_manifest_rows, duplicate_hashes, validate_manifest_rows, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GenImage manifest CSV.")
    parser.add_argument("--data_root", required=True, help="Root directory containing GenImage images.")
    parser.add_argument("--out", "--output", dest="out", required=True, help="Output manifest CSV path.")
    parser.add_argument("--split_mode", "--split_strategy", dest="split_mode", choices=["random", "generator_holdout"], default="random")
    parser.add_argument("--holdout_generators", default="", help="Comma-separated generator names for generator_holdout.")
    parser.add_argument("--max_samples_per_generator", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash_duplicates", action="store_true", help="Compute SHA256 duplicate report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_manifest_rows(args.data_root, args.split_mode, args.holdout_generators, args.max_samples_per_generator, args.seed)
    validate_manifest_rows([{k: str(v) for k, v in row.items()} for row in rows], strict=True)
    write_manifest(args.out, rows, columns=GENIMAGE_MANIFEST_COLUMNS)
    counts = Counter((row["generator"], row["split"], row["label"]) for row in rows)
    for key, value in sorted(counts.items()):
        print(f"generator={key[0]} split={key[1]} label={key[2]} count={value}")
    if args.hash_duplicates:
        duplicates = duplicate_hashes(rows)
        if duplicates:
            for digest, paths in duplicates.items():
                print(f"duplicate_hash={digest} paths={len(paths)}")
            raise SystemExit("Hash duplicate check failed")
    print(f"wrote {len(rows)} rows to {Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
