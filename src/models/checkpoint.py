from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch


REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "model_name",
    "input_dim",
    "hidden_dim",
    "threshold",
    "feature_type",
    "config_snapshot",
}


class CheckpointError(ValueError):
    pass


def save_checkpoint(
    path: str | Path,
    *,
    model_state_dict: Mapping[str, object],
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    threshold: float,
    feature_type: str,
    config_snapshot: Mapping[str, object],
    scaler_state: Mapping[str, object] | None = None,
) -> None:
    checkpoint: dict[str, object] = {
        "model_state_dict": dict(model_state_dict),
        "model_name": model_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "threshold": threshold,
        "feature_type": feature_type,
        "config_snapshot": dict(config_snapshot),
    }
    if scaler_state is not None:
        checkpoint["scaler_state"] = dict(scaler_state)

    validate_checkpoint(checkpoint)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_checkpoint(path: str | Path, *, expected_feature_type: str | None = None) -> dict[str, object]:
    checkpoint_path = Path(path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise CheckpointError("checkpoint file must contain a dictionary")
    checkpoint = cast(dict[str, object], loaded)
    validate_checkpoint(checkpoint, expected_feature_type=expected_feature_type)
    return checkpoint


def validate_checkpoint(checkpoint: Mapping[str, object], *, expected_feature_type: str | None = None) -> None:
    missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(checkpoint.keys()))
    if missing:
        raise CheckpointError(f"checkpoint missing required keys: {', '.join(missing)}")

    errors: list[str] = []
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        errors.append("model_state_dict must be a mapping")
    _require_non_empty_string(checkpoint["model_name"], "model_name", errors)
    _require_positive_int(checkpoint["input_dim"], "input_dim", errors)
    _require_positive_int(checkpoint["hidden_dim"], "hidden_dim", errors)
    _require_number(checkpoint["threshold"], "threshold", errors)
    feature_type = checkpoint["feature_type"]
    _require_non_empty_string(feature_type, "feature_type", errors)
    if expected_feature_type is not None and feature_type != expected_feature_type:
        errors.append(f"feature_type incompatible: expected {expected_feature_type}, got {feature_type}")
    if not isinstance(checkpoint["config_snapshot"], Mapping):
        errors.append("config_snapshot must be a mapping")
    if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None and not isinstance(checkpoint["scaler_state"], Mapping):
        errors.append("scaler_state must be a mapping when provided")

    if errors:
        raise CheckpointError("; ".join(errors))


def _require_non_empty_string(value: object, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{name} must be a non-empty string")


def _require_positive_int(value: object, name: str, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{name} must be a positive int")


def _require_number(value: object, name: str, errors: list[str]) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        errors.append(f"{name} must be a number")
