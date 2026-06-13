from __future__ import annotations

import argparse

# pyright: reportAny=false, reportUnusedCallResult=false
from pathlib import Path

from PIL import Image

from .make_split import assign_splits
from .validate_metadata import DATASET_COLUMNS, validate_metadata_rows, write_metadata


def make_dummy_dataset(
    num_real: int,
    num_fake: int,
    output_dir: str | Path,
    csv_path: str | Path,
    width: int = 224,
    height: int = 224,
    seed: int = 42,
) -> list[dict[str, str]]:
    root = Path(output_dir)
    rows: list[dict[str, str]] = []
    rows.extend(_make_class_rows(root, "real", num_real, width, height))
    rows.extend(_make_class_rows(root, "fake", num_fake, width, height))
    split_rows = assign_splits(rows, seed=seed)
    write_metadata(csv_path, split_rows)
    errors = validate_metadata_rows(split_rows, header=DATASET_COLUMNS, strict=False, check_files=True)
    if errors:
        raise ValueError("; ".join(errors))
    return split_rows


def _make_class_rows(root: Path, class_name: str, count: int, width: int, height: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    label = 0 if class_name == "real" else 1
    generator = "real_dummy" if class_name == "real" else "dummy_generator"
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        filename = f"{class_name}_{index:06d}.png"
        path = class_dir / filename
        _write_pattern_image(path, class_name=class_name, index=index, width=width, height=height)
        rows.append(
            {
                "image_id": f"dummy_{class_name}_{index:06d}",
                "filepath": path.as_posix(),
                "label": str(label),
                "class_name": class_name,
                "dataset": "DUMMY",
                "generator": generator,
                "split": "",
                "width": str(width),
                "height": str(height),
                "ext": "png",
            }
        )
    return rows


def _write_pattern_image(path: Path, class_name: str, index: int, width: int, height: int) -> None:
    base = (37, 91, 151) if class_name == "real" else (181, 73, 29)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    if pixels is None:
        raise RuntimeError("failed to allocate dummy image pixels")
    for y in range(height):
        for x in range(width):
            if class_name == "real":
                pixels[x, y] = (
                    (base[0] + x + index * 3) % 256,
                    (base[1] + y + index * 5) % 256,
                    (base[2] + x + y + index * 7) % 256,
                )
            else:
                checker = 64 if ((x // 16) + (y // 16) + index) % 2 else 0
                pixels[x, y] = (
                    (base[0] + checker + index * 11) % 256,
                    (base[1] + x * 2 + checker) % 256,
                    (base[2] + y * 2 + index * 13) % 256,
                )
    image.save(path, format="PNG")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic dummy image data and canonical dataset.csv metadata.")
    parser.add_argument("--num_real", type=int, required=True, help="Number of real dummy images")
    parser.add_argument("--num_fake", type=int, required=True, help="Number of fake dummy images")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root for dummy images")
    parser.add_argument("--csv", type=Path, required=True, help="Output dataset.csv path")
    parser.add_argument("--width", type=int, default=224, help="Dummy image width")
    parser.add_argument("--height", type=int, default=224, help="Dummy image height")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = make_dummy_dataset(
        num_real=args.num_real,
        num_fake=args.num_fake,
        output_dir=args.output_dir,
        csv_path=args.csv,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} dummy metadata rows: {args.csv}")
    print(f"Wrote dummy images under: {args.output_dir}")


if __name__ == "__main__":
    main()
