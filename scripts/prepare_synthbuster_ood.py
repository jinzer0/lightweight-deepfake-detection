from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import argparse
import hashlib
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, TypeVar

from PIL import Image

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional runtime nicety
    tqdm = None

from _path import ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.manifest import OK_STATUS, validate_manifest_rows, write_manifest  # noqa: E402
from src.data.validate_metadata import DATASET_COLUMNS, validate_metadata_rows, write_metadata  # noqa: E402


DEFAULT_ZIP_PATH = Path("data/raw/synthbuster.zip")
DEFAULT_OUTPUT_DIR = Path("data/synthbuster")
DEFAULT_METADATA_CSV = Path("data/metadata/synthbuster_ood_dataset.csv")
DEFAULT_MANIFEST_CSV = Path("outputs/synthbuster_ood/manifest.csv")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Synthbuster as an OOD test-only dataset for the current pipeline.")
    parser.add_argument("--zip_path", type=Path, default=DEFAULT_ZIP_PATH, help="Downloaded synthbuster.zip path.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where Synthbuster images are extracted.")
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV, help="Canonical dataset CSV output path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_CSV, help="Manifest-v1 CSV output path.")
    parser.add_argument("--clean", action="store_true", help="Remove existing prepared Synthbuster output before extracting.")
    parser.add_argument("--copy_docs", action="store_true", help="Also extract readme/licence/prompts files under output_dir/_docs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = prepare_synthbuster_ood(
            zip_path=args.zip_path,
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            manifest_path=args.manifest,
            clean=args.clean,
            copy_docs=args.copy_docs,
        )
        _print_counts(rows)
        print(f"wrote metadata: {args.metadata_csv}")
        print(f"wrote manifest: {args.manifest}")
        return 0
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Synthbuster OOD preparation failed: {error}", file=sys.stderr)
        return 1


def prepare_synthbuster_ood(
    *,
    zip_path: Path,
    output_dir: Path,
    metadata_csv: Path,
    manifest_path: Path,
    clean: bool = False,
    copy_docs: bool = False,
) -> list[dict[str, str]]:
    zip_path = zip_path.expanduser()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Synthbuster zip not found: {zip_path}")
    if clean and output_dir.exists():
        _clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".synthbuster_ood_root").write_text("created by prepare_synthbuster_ood.py\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path) as archive:
        image_infos = _image_infos(archive)
        if not image_infos:
            raise ValueError(f"No image files found in {zip_path}")
        metadata_rows, manifest_rows = _extract_images(archive, image_infos, output_dir=output_dir)
        if copy_docs:
            _extract_docs(archive, output_dir=output_dir)

    write_metadata(metadata_csv, metadata_rows)
    metadata_errors = validate_metadata_rows(metadata_rows, header=DATASET_COLUMNS, strict=False, check_files=True)
    if metadata_errors:
        raise ValueError("; ".join(metadata_errors))

    write_manifest(manifest_path, manifest_rows)
    validate_manifest_rows([{key: str(value) for key, value in row.items()} for row in manifest_rows], strict=True)
    return metadata_rows


def _image_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = Path(info.filename)
        if len(path.parts) >= 3 and path.parts[0] == "synthbuster" and path.suffix.lower() in IMAGE_SUFFIXES:
            infos.append(info)
    return sorted(infos, key=lambda item: item.filename)


def _extract_images(
    archive: zipfile.ZipFile,
    image_infos: list[zipfile.ZipInfo],
    *,
    output_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    metadata_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    root = output_dir.resolve()
    for info in _progress(image_infos, desc="Extracting Synthbuster images", total=len(image_infos), unit="image"):
        archive_path = Path(info.filename)
        generator = archive_path.parts[1]
        filename = archive_path.name
        safe_generator = _safe_name(generator)
        rel_path = Path("test") / safe_generator / "fake" / filename
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        width, height = _image_size(output_path)
        digest = _sha256(output_path)
        sample_id = f"synthbuster_{safe_generator}_fake_{Path(filename).stem}_{digest[:12]}"
        stat = output_path.stat()
        metadata_rows.append(
            {
                "image_id": sample_id,
                "filepath": output_path.resolve().as_posix(),
                "label": "1",
                "class_name": "fake",
                "dataset": "Synthbuster-OOD",
                "generator": generator,
                "split": "test",
                "width": str(width),
                "height": str(height),
                "ext": output_path.suffix.lower().lstrip("."),
            }
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "base_sample_id": sample_id,
                "rel_path": rel_path.as_posix(),
                "root": root.as_posix(),
                "label": "1",
                "class_name": "fake",
                "source": f"Synthbuster:{generator}",
                "source_split": "ood",
                "split": "test",
                "width": str(width),
                "height": str(height),
                "sha256": digest,
                "file_size": str(stat.st_size),
                "mtime": str(stat.st_mtime_ns),
                "status": OK_STATUS,
            }
        )
    metadata_rows.sort(key=lambda item: item["image_id"])
    manifest_rows.sort(key=lambda item: str(item["sample_id"]))
    return metadata_rows, manifest_rows


def _extract_docs(archive: zipfile.ZipFile, *, output_dir: Path) -> None:
    docs_dir = output_dir / "_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = Path(info.filename)
        if len(path.parts) == 2 and path.parts[0] == "synthbuster" and path.suffix.lower() not in IMAGE_SUFFIXES:
            target = docs_dir / path.name
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
        return image.size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved == Path("/") or len(resolved.parts) < 4:
        raise ValueError(f"refusing --clean on broad output_dir: {resolved}")
    marker = resolved / ".synthbuster_ood_root"
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("created by prepare_synthbuster_ood.py\n", encoding="utf-8")
        raise ValueError(f"created cleanup marker at {marker}; rerun --clean to delete prepared Synthbuster files")
    shutil.rmtree(resolved)


def _safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_") or "unknown"


def _progress(iterable: Iterable[T], *, desc: str, unit: str, total: int | None = None) -> Iterable[T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)


def _print_counts(rows: list[dict[str, str]]) -> None:
    print("Prepared Synthbuster OOD rows:")
    print(f"  total: {len(rows)}")
    by_generator = Counter(row["generator"] for row in rows)
    for generator in sorted(by_generator):
        print(f"  {generator}: {by_generator[generator]}")


if __name__ == "__main__":
    raise SystemExit(main())
