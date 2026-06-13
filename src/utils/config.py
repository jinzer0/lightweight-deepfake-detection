from __future__ import annotations

from pathlib import Path
from typing import cast
import warnings

import yaml


def load_config(config_path: str) -> dict[str, object]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj)  # pyright: ignore[reportAny]
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse YAML config {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a mapping: {path}")

    return cast(dict[str, object], payload)


def resolve_device(config: dict[str, object]) -> str:
    requested_device = "cpu"
    project = config.get("project")
    if isinstance(project, dict):
        project_config = cast(dict[str, object], project)
        requested_device = str(project_config.get("device", "cpu"))

    if requested_device != "cuda":
        return requested_device

    try:
        import torch
    except ModuleNotFoundError:
        warnings.warn("CUDA is unavailable; using cpu.", RuntimeWarning, stacklevel=2)
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    warnings.warn("CUDA is unavailable; using cpu.", RuntimeWarning, stacklevel=2)
    return "cpu"
