from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from pathlib import Path

import pytest

from src.inference import ImagePredictor, PredictorArtifactError, UnsupportedPredictorArtifactError
from src.train.frequency_lr import train_classifier
from tests.test_training_matrix import _write_inputs


def test_frequency_lr_predictor_loads_and_predicts_probability(tiny_png: Path, tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "frequency_lr",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )

    predictor = ImagePredictor.from_experiment_dir(result.output_dir)
    prediction = predictor.predict(tiny_png, threshold=0.25)

    assert 0.0 <= prediction.prob_fake <= 1.0
    assert prediction.artifact_threshold == 0.5
    assert prediction.effective_threshold == 0.25
    assert prediction.threshold == 0.25
    assert prediction.pred_label in {0, 1}
    assert prediction.label_text in {"real", "fake"}
    assert prediction.limitations["experimental_warning"].startswith("This detector is an experimental model")


@pytest.mark.parametrize("file_name", ["config.yaml", "model.joblib", "scaler.joblib"])
def test_predictor_rejects_missing_required_artifact(file_name: str, tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / f"missing_{file_name}",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    (result.output_dir / file_name).unlink()

    with pytest.raises(PredictorArtifactError, match="missing required artifact"):
        ImagePredictor.from_experiment_dir(result.output_dir)


def test_predictor_rejects_score_only_linear_svm_artifact(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "frequency_svm",
        mode="frequency_only",
        classifier="linear_svm",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=500,
        verify_reload=False,
    )

    with pytest.raises(UnsupportedPredictorArtifactError, match="score-only Linear SVM"):
        ImagePredictor.from_experiment_dir(result.output_dir)
