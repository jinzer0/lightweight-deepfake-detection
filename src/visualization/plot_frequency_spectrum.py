from __future__ import annotations

import argparse
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from src.utils.config import load_config
from src.visualization.radial_spectrum import radial_spectrum_from_image, save_radial_spectrum_plot
from src.visualization.spectrum import SUPPORTED_METHODS, save_spectrum_image


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    frequency_settings = config.get("frequency", {})
    if not isinstance(frequency_settings, dict):
        frequency_settings = {}

    method = args.method or str(frequency_settings.get("method", "dct"))
    image_size = int(args.image_size or frequency_settings.get("image_size", 512))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or f"{_safe_stem(Path(args.image))}_{uuid.uuid4().hex[:12]}"
    spectrum_path = output_dir / f"{prefix}_spectrum.png"
    radial_path = output_dir / f"{prefix}_radial.png"

    radial_spectrum = radial_spectrum_from_image(
        args.image,
        config,
        method=method,
        image_size=image_size,
        radial_bins=args.radial_bins,
    )
    saved_spectrum = save_spectrum_image(args.image, spectrum_path.as_posix(), method=method, image_size=image_size)
    saved_radial = save_radial_spectrum_plot(
        radial_spectrum,
        radial_path.as_posix(),
        method=method,
        log_scale=bool(frequency_settings.get("log_scale", False)),
        normalize_feature=bool(frequency_settings.get("normalize_feature", False)),
    )

    print(f"spectrum_path={saved_spectrum}")
    print(f"radial_spectrum_path={saved_radial}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save DCT/FFT spectrum and matching radial spectrum plots for one image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config with frequency visualization settings.")
    parser.add_argument("--output-dir", default="outputs/plots", help="Directory for generated PNG plots.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, help="Override frequency.method from config.")
    parser.add_argument("--image-size", type=int, help="Override frequency.image_size from config.")
    parser.add_argument("--radial-bins", type=int, help="Override frequency.radial_bins from config.")
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to <image_stem>_<token>.")
    return parser.parse_args(argv)


def _safe_stem(image_path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("._")
    return cleaned or "image"


if __name__ == "__main__":
    raise SystemExit(main())
