from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

import argparse
import sys
from pathlib import Path

import numpy as np

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.manifest import OK_STATUS, ManifestValidationError, read_manifest, validate_manifest_rows  # noqa: E402
from src.features.cache import build_metadata, create_feature_cache, hash_manifest_rows, write_feature_cache  # noqa: E402
from src.features.frequency import (  # noqa: E402
    DCT_BACKEND,
    DCT_POLICY,
    DEFAULT_FFT_EPSILON,
    DEFAULT_RADIAL_BINS,
    FEATURE_DTYPE,
    FrequencyFeatureConfig,
    extract_frequency_features,
)
from src.utils.image_io import DEFAULT_FREQUENCY_IMAGE_SIZE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract compact FFT/DCT frequency features into a feature_cache_v1 .pt file.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest v1 CSV.")
    parser.add_argument("--output_cache", type=Path, required=True, help="Output .pt feature cache path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional maximum number of manifest-order samples to extract.")
    parser.add_argument("--image_size", type=int, default=DEFAULT_FREQUENCY_IMAGE_SIZE, help="Frequency preprocessing size.")
    parser.add_argument("--radial_bins", type=int, default=DEFAULT_RADIAL_BINS, help="Number of centered FFT radial spectrum bins.")
    parser.add_argument("--fft_epsilon", type=float, default=DEFAULT_FFT_EPSILON, help="Epsilon used for log(abs(fft) + epsilon).")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in cache metadata; extraction is deterministic.")
    parser.add_argument("--skip_non_ok", action="store_true", help="Skip manifest rows whose status is not ok instead of failing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_manifest(args.manifest)
        selected_rows = _select_rows(rows, max_samples=args.max_samples, skip_non_ok=args.skip_non_ok)
        validate_manifest_rows(selected_rows, strict=True)

        config = FrequencyFeatureConfig(image_size=args.image_size, radial_bins=args.radial_bins, fft_epsilon=args.fft_epsilon)
        feature_rows: list[np.ndarray] = []
        for row_index, row in enumerate(selected_rows, start=1):
            image_path = _resolve_image_path(row)
            try:
                feature_rows.append(
                    extract_frequency_features(
                        image_path,
                        image_size=args.image_size,
                        radial_bins=args.radial_bins,
                        fft_epsilon=args.fft_epsilon,
                    )
                )
            except Exception as exc:
                sample_id = row.get("sample_id", f"row-{row_index}")
                raise RuntimeError(f"failed to extract sample {sample_id} at {image_path}: {exc}") from exc

        features = np.vstack(feature_rows).astype(FEATURE_DTYPE, copy=False)
        metadata = build_metadata(
            feature_dim=int(features.shape[1]),
            dtype=str(np.dtype(FEATURE_DTYPE).name),
            normalization="raw_unscaled",
            seed=args.seed,
            extra={
                "image_size": int(args.image_size),
                "radial_bins": int(args.radial_bins),
                "fft_epsilon": float(args.fft_epsilon),
                "dct_policy": DCT_POLICY,
                "dct_backend": DCT_BACKEND,
                "feature_order": "fft_radial_64_then_fft_summary_then_whole_dct_then_block_dct",
            },
        )
        cache = create_feature_cache(
            manifest_rows=selected_rows,
            feature_type="frequency",
            feature_config=config.as_dict(),
            features=features,
            metadata=metadata,
            manifest_hash=hash_manifest_rows(selected_rows),
        )
        write_feature_cache(cache, args.output_cache)
        print(f"wrote frequency cache: {args.output_cache}")
        print(f"samples: {features.shape[0]}")
        print(f"feature_dim: {features.shape[1]}")
        return 0
    except (ManifestValidationError, RuntimeError, ValueError) as error:
        print(f"frequency extraction failed: {error}", file=sys.stderr)
        return 1


def _select_rows(rows: list[dict[str, str]], *, max_samples: int | None, skip_non_ok: bool) -> list[dict[str, str]]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max_samples must be positive when provided")
    if not rows:
        raise ValueError("manifest has no rows")

    if skip_non_ok:
        selected: list[dict[str, str]] = []
        skipped: list[str] = []
        for row in rows:
            if row.get("status", "") != OK_STATUS:
                skipped.append(f"{row.get('sample_id', '')}:{row.get('status', '')}")
                continue
            selected.append(row)
            if max_samples is not None and len(selected) >= max_samples:
                break
        if skipped:
            print(f"skipped non-ok rows: {', '.join(skipped[:10])}")
        if not selected:
            raise ValueError("no ok manifest rows selected")
        return selected

    selected = rows[:max_samples] if max_samples is not None else rows
    non_ok = [f"{row.get('sample_id', '')}:{row.get('status', '')}" for row in selected if row.get("status", "") != OK_STATUS]
    if non_ok:
        raise ValueError(f"manifest contains non-ok rows; pass --skip_non_ok to skip them: {', '.join(non_ok[:10])}")
    return selected


def _resolve_image_path(row: dict[str, str]) -> Path:
    rel_path = Path(row.get("rel_path", ""))
    if rel_path.is_absolute():
        return rel_path
    root = row.get("root", "")
    if not root:
        raise ValueError(f"manifest row {row.get('sample_id', '')} has empty root")
    return Path(root) / rel_path


if __name__ == "__main__":
    raise SystemExit(main())
