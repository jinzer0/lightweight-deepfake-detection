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
from src.features.clip import (  # noqa: E402
    CLIP_FEATURE_DIM,
    CLIP_FEATURE_DTYPE,
    CLIP_NORMALIZATION,
    DEFAULT_CLIP_MODEL_ID,
    ClipDependencyError,
    ClipFeatureConfig,
    CudaUnavailableError,
    extract_clip_features,
)

DEFAULT_BLOCKER_EVIDENCE_PATH = Path(".sisyphus/evidence/task-8-clip-smoke.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen openCLIP ViT-B/32 image embeddings into a feature_cache_v1 .pt file.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest v1 CSV.")
    parser.add_argument("--output_cache", type=Path, required=True, help="Output .pt feature cache path.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional maximum number of manifest-order samples to extract.")
    parser.add_argument("--batch_size", type=int, default=32, help="CLIP inference batch size.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device. auto prefers CUDA when available.")
    parser.add_argument("--model_id", default=DEFAULT_CLIP_MODEL_ID, help="openCLIP model identifier.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in cache metadata; extraction is deterministic.")
    parser.add_argument("--skip_non_ok", action="store_true", help="Skip manifest rows whose status is not ok instead of failing.")
    parser.add_argument("--smoke", action="store_true", help="Treat model download/runtime failures as explicit smoke blockers instead of uncaught tracebacks.")
    parser.add_argument("--write_blocker", type=Path, default=None, help="Write external smoke blocker evidence to this path on model/runtime failure.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = read_manifest(args.manifest)
        selected_rows = _select_rows(rows, max_samples=args.max_samples, skip_non_ok=args.skip_non_ok)
        validate_manifest_rows(selected_rows, strict=True)

        config = ClipFeatureConfig(model_id=args.model_id, batch_size=args.batch_size, device=args.device, normalize=True)
        image_paths = [_resolve_image_path(row) for row in selected_rows]
        try:
            features, runtime = extract_clip_features(
                image_paths,
                model_id=args.model_id,
                batch_size=args.batch_size,
                device=args.device,
                normalize=True,
            )
        except CudaUnavailableError:
            raise
        except Exception as exc:
            if args.smoke or args.write_blocker is not None:
                _write_blocker(args.write_blocker or DEFAULT_BLOCKER_EVIDENCE_PATH, args=args, error=exc)
                print(f"CLIP smoke blocked by external model/runtime failure: {exc}", file=sys.stderr)
                return 2
            raise RuntimeError(f"failed to extract CLIP features: {exc}") from exc

        if int(features.shape[1]) != CLIP_FEATURE_DIM:
            raise ValueError(f"CLIP feature dimension must be {CLIP_FEATURE_DIM}, got {features.shape[1]}")

        metadata = build_metadata(
            feature_dim=int(features.shape[1]),
            dtype=str(np.dtype(CLIP_FEATURE_DTYPE).name),
            normalization=CLIP_NORMALIZATION,
            seed=args.seed,
            extra={
                "model_name": args.model_id,
                "preprocess_hash": runtime["preprocess_hash"],
                "device": runtime["device"],
                "batch_size": int(args.batch_size),
            },
        )
        cache = create_feature_cache(
            manifest_rows=selected_rows,
            feature_type="clip",
            feature_config=config.as_dict(),
            features=features,
            metadata=metadata,
            manifest_hash=hash_manifest_rows(selected_rows),
        )
        write_feature_cache(cache, args.output_cache)
        print(f"wrote CLIP cache: {args.output_cache}")
        print(f"samples: {features.shape[0]}")
        print(f"feature_dim: {features.shape[1]}")
        print(f"device: {runtime['device']}")
        return 0
    except (ManifestValidationError, RuntimeError, ValueError, ClipDependencyError) as error:
        print(f"CLIP extraction failed: {error}", file=sys.stderr)
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


def _write_blocker(path: Path, *, args: argparse.Namespace, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    message = (
        "CLIP CPU smoke external blocker\n"
        f"model_id: {args.model_id}\n"
        f"device: {args.device}\n"
        f"manifest: {args.manifest}\n"
        f"output_cache: {args.output_cache}\n"
        f"error_type: {type(error).__name__}\n"
        f"error: {error}\n"
    )
    path.write_text(message, encoding="utf-8")
    print(f"wrote CLIP smoke blocker evidence: {path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
