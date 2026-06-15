from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable, TypeVar, cast

from PIL import Image

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional runtime nicety until requirements are installed
    tqdm = None

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.manifest import OK_STATUS, validate_manifest_rows, write_manifest  # noqa: E402
from src.data.validate_metadata import DATASET_COLUMNS, validate_metadata_rows, write_metadata  # noqa: E402


DEFAULT_DATASET_ID = "TheKernel01/Tiny-GenImage"
DEFAULT_IMAGE_SIZE = 512
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize all Tiny-GenImage rows with source train as train and source validation split 50/50 into val/test.")
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset id loaded with datasets.load_dataset.")
    parser.add_argument("--image_root", type=Path, default=Path("data/genimage_tiny_full_512"), help="Output image root for materialized 512x512 JPG files.")
    parser.add_argument("--metadata_csv", type=Path, default=Path("data/metadata/genimage_tiny_full_dataset.csv"), help="Canonical dataset.csv output path.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/genimage_tiny_full/manifest.csv"), help="Manifest-v1 output path for CUDA fine-tuning.")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Materialized square image size. Default is 512.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--clean", action="store_true", help="Remove existing materialized JPG files under image_root before writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = prepare_tiny_genimage(
            dataset_id=args.dataset_id,
            image_root=args.image_root,
            metadata_csv=args.metadata_csv,
            manifest_path=args.manifest,
            image_size=args.image_size,
            seed=args.seed,
            clean=args.clean,
        )
        _print_counts(rows)
        print(f"wrote metadata: {args.metadata_csv}")
        print(f"wrote manifest: {args.manifest}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Tiny-GenImage preparation failed: {error}", file=sys.stderr)
        return 1


def prepare_tiny_genimage(
    *,
    dataset_id: str,
    image_root: Path,
    metadata_csv: Path,
    manifest_path: Path,
    image_size: int,
    seed: int,
    clean: bool,
) -> list[dict[str, str]]:
    if image_size <= 0:
        raise ValueError("--image_size must be positive")
    rows = _load_rows(dataset_id)
    split_rows = assign_tiny_genimage_default_splits(rows, seed=seed)
    if clean and image_root.exists():
        resolved_root = image_root.resolve()
        if resolved_root == Path("/") or len(resolved_root.parts) < 4:
            raise ValueError(f"refusing --clean on broad image_root: {resolved_root}")
        marker = resolved_root / ".genimage_subset_root"
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("created by prepare_genimage_subset.py\n", encoding="utf-8")
            raise ValueError(f"created cleanup marker at {marker}; rerun --clean to delete generated jpg files")
        jpg_paths = list(resolved_root.rglob("*.jpg"))
        for path in _progress(jpg_paths, desc="Cleaning existing JPGs", unit="file"):
            path.unlink()
    image_root.mkdir(parents=True, exist_ok=True)
    (image_root / ".genimage_subset_root").write_text("created by prepare_genimage_subset.py\n", encoding="utf-8")
    metadata_rows, manifest_rows = _materialize_rows(split_rows, image_root=image_root, image_size=image_size)
    metadata_rows, manifest_rows, dropped_rows = _drop_cross_split_duplicate_hashes(metadata_rows, manifest_rows)
    if dropped_rows:
        _print_dropped_duplicates(dropped_rows)
    write_metadata(metadata_csv, metadata_rows)
    metadata_errors = validate_metadata_rows(metadata_rows, header=DATASET_COLUMNS, strict=False, check_files=True)
    if metadata_errors:
        raise ValueError("; ".join(metadata_errors))
    write_manifest(manifest_path, manifest_rows)
    validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in manifest_rows], strict=True)
    return metadata_rows


def _load_rows(dataset_id: str) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is required; install it with `python -m pip install datasets` or `pip install -r requirements.txt`") from exc
    loaded = load_dataset(dataset_id)
    dataset = cast(Mapping[str, Any], loaded)
    rows: list[dict[str, Any]] = []
    first_split = dataset["train"] if isinstance(loaded, DatasetDict) and "train" in dataset else next(iter(dataset.values()))
    features = first_split.features
    generator_feature = features["generator"]
    label_feature = features["label"]
    for source_split, split_data in dataset.items():
        total = len(split_data) if hasattr(split_data, "__len__") else None
        for index, raw_item in enumerate(_progress(split_data, desc=f"Loading {source_split} rows", total=total, unit="row")):
            item = dict(cast(dict[str, Any], raw_item))
            label_id = int(item["label"])
            generator_id = int(item["generator"])
            rows.append(
                {
                    "image": item["image"],
                    "label": label_id,
                    "class_name": _label_name(label_feature, label_id),
                    "generator": _generator_name(generator_feature, generator_id, label_id),
                    "source_split": str(source_split),
                    "source_index": index,
                }
            )
    if not rows:
        raise ValueError(f"dataset {dataset_id} produced no rows")
    return rows


def _label_name(label_feature: Any, label_id: int) -> str:
    name = str(label_feature.int2str(label_id)).lower() if hasattr(label_feature, "int2str") else str(label_id)
    if name == "real" or label_id == 0:
        return "real"
    if name == "fake" or label_id == 1:
        return "fake"
    raise ValueError(f"unsupported label {label_id}: {name}")


def _generator_name(generator_feature: Any, generator_id: int, label_id: int) -> str:
    if label_id == 0:
        return "Real"
    if hasattr(generator_feature, "int2str"):
        return str(generator_feature.int2str(generator_id))
    return str(generator_id)


def assign_tiny_genimage_default_splits(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Use Tiny-GenImage source train for train and split source validation into val/test.

    The current default is 28,000 train rows and a deterministic 3,500/3,500
    split of the 7,000 source validation rows for validation and test.
    """
    output = [dict(row) for row in rows]
    validation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    randomizer = random.Random(seed)
    for row in output:
        source_split = str(row.get("source_split", "")).lower()
        if source_split == "train":
            row["split"] = "train"
        elif source_split in {"validation", "val", "valid", "test"}:
            validation_groups[(str(row["class_name"]), str(row["generator"]))].append(row)
        else:
            row["split"] = "train"
    for group_rows in validation_groups.values():
        randomizer.shuffle(group_rows)
        val_count = len(group_rows) // 2
        for index, row in enumerate(group_rows):
            row["split"] = "val" if index < val_count else "test"
    return output


def _assign_70_20_10_splits(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    return assign_tiny_genimage_default_splits(rows, seed=seed)


def _split_names(count: int) -> list[str]:
    train_count = int(count * 0.80)
    val_count = int(count * 0.10)
    if count >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    return ["train"] * train_count + ["val"] * val_count + ["test"] * (count - train_count - val_count)

def _materialize_rows(rows: list[dict[str, Any]], *, image_root: Path, image_size: int) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    manifest_root = image_root.resolve()
    for row in _progress(rows, desc="Materializing images", total=len(rows), unit="image"):
        class_name = str(row["class_name"])
        generator = str(row["generator"])
        split = str(row["split"])
        safe_generator = _safe_name(generator)
        key = (split, class_name, generator)
        counters[key] += 1
        rel_path = Path(split) / safe_generator / class_name / f"{class_name}_{safe_generator}_{counters[key]:06d}.jpg"
        output_path = image_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = row["image"]
        pil_image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        resized = pil_image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        width, height = resized.size
        resized.save(output_path, format="JPEG", quality=95)
        digest = _sha256(output_path)
        image_id = f"tiny_genimage_{safe_generator}_{class_name}_{split}_{counters[key]:06d}_{digest[:12]}"
        label = "0" if class_name == "real" else "1"
        metadata_rows.append({"image_id": image_id, "filepath": output_path.resolve().as_posix(), "label": label, "class_name": class_name, "dataset": "Tiny-GenImage", "generator": generator, "split": split, "width": str(width), "height": str(height), "ext": "jpg"})
        stat = output_path.stat()
        manifest_rows.append({"sample_id": image_id, "base_sample_id": image_id, "rel_path": rel_path.as_posix(), "root": manifest_root.as_posix(), "label": label, "class_name": class_name, "source": f"Tiny-GenImage:{generator}", "source_split": str(row["source_split"]), "split": split, "width": str(width), "height": str(height), "sha256": digest, "file_size": str(stat.st_size), "mtime": str(stat.st_mtime_ns), "status": OK_STATUS})
    return sorted(metadata_rows, key=lambda item: item["image_id"]), sorted(manifest_rows, key=lambda item: str(item["sample_id"]))


def _drop_cross_split_duplicate_hashes(
    metadata_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, object]],
) -> tuple[list[dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    rows_by_digest: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        digest = str(row.get("sha256", ""))
        if digest:
            rows_by_digest[digest].append(row)

    split_priority = {"train": 0, "val": 1, "test": 2}
    dropped_rows: list[dict[str, object]] = []
    dropped_sample_ids: set[str] = set()
    for digest_rows in rows_by_digest.values():
        if len({str(row.get("split", "")) for row in digest_rows}) <= 1:
            continue
        keep_row = min(
            digest_rows,
            key=lambda row: (
                split_priority.get(str(row.get("split", "")), 99),
                str(row.get("source_split", "")),
                str(row.get("rel_path", "")),
            ),
        )
        keep_sample_id = str(keep_row.get("sample_id", ""))
        for row in digest_rows:
            sample_id = str(row.get("sample_id", ""))
            if sample_id != keep_sample_id:
                dropped_rows.append(row)
                dropped_sample_ids.add(sample_id)
                _unlink_materialized_file(row)

    if not dropped_rows:
        return metadata_rows, manifest_rows, []

    filtered_metadata_rows = [row for row in metadata_rows if row["image_id"] not in dropped_sample_ids]
    filtered_manifest_rows = [row for row in manifest_rows if str(row.get("sample_id", "")) not in dropped_sample_ids]
    return filtered_metadata_rows, filtered_manifest_rows, sorted(dropped_rows, key=lambda row: str(row.get("sample_id", "")))


def _unlink_materialized_file(row: dict[str, object]) -> None:
    root_text = str(row.get("root", ""))
    rel_text = str(row.get("rel_path", ""))
    if not root_text or not rel_text:
        return
    rel_path = Path(rel_text)
    if rel_path.is_absolute():
        return
    (Path(root_text) / rel_path).unlink(missing_ok=True)


def _print_dropped_duplicates(dropped_rows: list[dict[str, object]]) -> None:
    print(f"dropped {len(dropped_rows)} duplicate materialized row(s) to prevent sha256 leakage across splits")
    for row in dropped_rows[:10]:
        digest_prefix = str(row.get("sha256", ""))[:12]
        print(f"dropped duplicate: split={row.get('split')} sha256={digest_prefix} rel_path={row.get('rel_path')}")
    if len(dropped_rows) > 10:
        print(f"... {len(dropped_rows) - 10} more duplicate row(s) dropped")


def _progress(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def _print_counts(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["split"], row["class_name"], row["generator"])] += 1
    print("split,class_name,generator,count")
    for key in sorted(counts):
        print(f"{key[0]},{key[1]},{key[2]},{counts[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
