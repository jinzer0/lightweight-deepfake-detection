from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportUnusedCallResult=false

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.data.make_dummy_dataset import make_dummy_dataset
from src.data.dataset import ImageMetadataDataset
from src.data.transforms import get_eval_transform
from src.features.cache_features import (
    NpyFeatureCacheError,
    cache_frequency_split,
    cache_paths,
    load_feature_cache,
    validate_feature_cache_arrays,
)
from src.features.clip_features import ClipModelLoadError, load_clip_model
from src.features.frequency_features import extract_frequency_feature


def test_frequency_cache_writes_loads_and_preserves_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)

    features, labels, meta = cache_frequency_split(config, split="train")
    loaded_features, loaded_labels, loaded_meta = load_feature_cache(config, feature_type="frequency", split="train")

    paths = cache_paths(config, feature_type="frequency", split="train")
    assert paths.features.is_file()
    assert paths.labels.is_file()
    assert paths.meta.is_file()
    assert features.dtype == np.float32
    assert labels.dtype == np.int64
    assert loaded_features.shape == features.shape
    assert loaded_labels.tolist() == labels.tolist()
    assert loaded_meta["image_id"].tolist() == meta["image_id"].tolist()
    assert {"image_id", "filepath", "label", "class_name", "dataset", "generator", "split", "width", "height", "ext"}.issubset(loaded_meta.columns)
    assert set(loaded_meta["split"]) == {"train"}


def test_frequency_cache_matches_raw_image_extraction_not_clip_tensor(tmp_path: Path) -> None:
    config = _config(tmp_path)

    features, _labels, meta = cache_frequency_split(config, split="train")
    first_path = Path(str(meta.iloc[0]["filepath"]))
    raw_feature = extract_frequency_feature(first_path, config)

    paths_config = cast(dict[str, object], config["paths"])
    data_config = cast(dict[str, object], config["data"])
    clip_dataset = ImageMetadataDataset(
        str(paths_config["dataset_csv"]),
        split="train",
        transform=get_eval_transform(int(cast(int, data_config["image_size"]))),
        return_metadata=True,
    )
    clip_tensor, _label, _metadata = cast(tuple[object, int, dict[str, str]], clip_dataset[0])
    clip_tensor_feature = extract_frequency_feature(clip_tensor, config)

    np.testing.assert_allclose(features[0], raw_feature, rtol=1e-6, atol=1e-6)
    assert not np.allclose(raw_feature, clip_tensor_feature, rtol=1e-6, atol=1e-6)


def test_load_feature_cache_rejects_split_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _ = cache_frequency_split(config, split="train")
    paths = cache_paths(config, feature_type="frequency", split="train")
    meta = pd.read_csv(paths.meta)
    meta.loc[0, "split"] = "val"
    meta.to_csv(paths.meta, index=False)

    with pytest.raises(NpyFeatureCacheError, match="metadata split values.*train"):
        load_feature_cache(config, feature_type="frequency", split="train")


def test_validate_feature_cache_arrays_rejects_duplicate_image_ids_and_label_mismatch() -> None:
    features = np.ones((2, 3), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    meta = pd.DataFrame(
        {
            "image_id": ["same", "same"],
            "filepath": ["a.png", "b.png"],
            "label": [1, 1],
            "class_name": ["real", "fake"],
            "dataset": ["DUMMY", "DUMMY"],
            "generator": ["real_dummy", "dummy_generator"],
            "split": ["train", "train"],
            "width": ["16", "16"],
            "height": ["16", "16"],
            "ext": ["png", "png"],
        }
    )

    with pytest.raises(NpyFeatureCacheError) as error:
        validate_feature_cache_arrays(features, labels, meta, feature_type="frequency", split="train", context="unit-cache")

    message = str(error.value)
    assert "duplicate" in message
    assert "labels do not match metadata label" in message
    assert "unit-cache" in message


def test_invalid_split_fails_before_cache_reuse(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(NpyFeatureCacheError, match="Invalid split"):
        load_feature_cache(config, feature_type="frequency", split="dev")


def test_clip_load_failure_message_is_optional_smoke_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "open_clip", None)

    with pytest.raises(ClipModelLoadError) as error:
        load_clip_model({"clip": {"model_name": "ViT-B-32", "pretrained": "openai"}}, device="cpu")

    message = str(error.value)
    assert "open_clip_torch" in message
    assert "offline optional-smoke" in message


def test_cache_features_cli_missing_config_is_concise(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing-config.yaml"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.features.cache_features",
            "--config",
            missing_config.as_posix(),
            "--feature_type",
            "frequency",
            "--split",
            "train",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert "Feature cache failed clearly" in output
    assert missing_config.as_posix() in output


def test_cache_features_cli_missing_dataset_csv_is_concise(tmp_path: Path) -> None:
    missing_csv = tmp_path / "metadata" / "dataset.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "paths": {"dataset_csv": missing_csv.as_posix(), "feature_dir": (tmp_path / "features").as_posix()},
                "frequency": {"method": "dct", "image_size": 24, "radial_bins": 8, "log_scale": True, "normalize_feature": True},
                "project": {"device": "cpu", "seed": 42},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.features.cache_features",
            "--config",
            config_path.as_posix(),
            "--feature_type",
            "frequency",
            "--split",
            "train",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert "Feature cache failed clearly" in output
    assert missing_csv.as_posix() in output


def _config(tmp_path: Path) -> dict[str, object]:
    csv_path = tmp_path / "metadata" / "dataset.csv"
    _ = make_dummy_dataset(num_real=6, num_fake=6, output_dir=tmp_path / "images", csv_path=csv_path, width=32, height=24)
    return {
        "paths": {"dataset_csv": csv_path.as_posix(), "feature_dir": (tmp_path / "features").as_posix()},
        "data": {"image_size": 32, "batch_size": 4, "num_workers": 0},
        "frequency": {"method": "dct", "image_size": 32, "radial_bins": 8, "log_scale": True, "normalize_feature": True},
        "clip": {"model_name": "ViT-B-32", "pretrained": "openai", "normalize_feature": True, "freeze": True},
        "project": {"device": "cpu", "seed": 42},
    }
