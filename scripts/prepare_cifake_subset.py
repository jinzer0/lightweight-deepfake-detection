from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportAny=false, reportUnknownArgumentType=false

import argparse
import sys
from pathlib import Path

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.cifake import generate_and_write_manifest  # noqa: E402
from src.data.manifest import ManifestValidationError, validate_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate a deterministic manifest v1 for a local CIFAKE-style image folder."
    )
    parser.add_argument("--data_root", type=Path, help="Local CIFAKE-style root containing real/fake folders.")
    parser.add_argument("--output_manifest", type=Path, help="CSV manifest path to write.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for class-balanced project split assignment.")
    parser.add_argument("--num_real", type=int, default=None, help="Optional maximum number of real samples.")
    parser.add_argument("--num_fake", type=int, default=None, help="Optional maximum number of fake samples.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Fail on validation errors.")
    parser.add_argument("--validate_manifest", type=Path, help="Validate an existing manifest instead of generating one.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.validate_manifest is not None:
            errors = validate_manifest(args.validate_manifest, strict=args.strict)
            _print_validation_result(errors)
            return 0

        if args.data_root is None or args.output_manifest is None:
            raise SystemExit("--data_root and --output_manifest are required unless --validate_manifest is used")

        errors = generate_and_write_manifest(
            data_root=args.data_root,
            output_manifest=args.output_manifest,
            seed=args.seed,
            num_real=args.num_real,
            num_fake=args.num_fake,
            strict=args.strict,
        )
        _print_validation_result(errors)
        print(f"wrote manifest: {args.output_manifest}")
        return 0
    except ManifestValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1


def _print_validation_result(errors: list[str]) -> None:
    if errors:
        print("validation warnings:")
        for error in errors:
            print(f"- {error}")
    else:
        print("validation passed")


if __name__ == "__main__":
    raise SystemExit(main())
