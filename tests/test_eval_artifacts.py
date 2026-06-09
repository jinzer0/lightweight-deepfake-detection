from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

import csv
import json
import shutil
from pathlib import Path

import pytest

from src.eval import ArtifactValidationError, evaluate_experiment, validate_experiment_artifacts
from src.train.frequency_lr import train_classifier
from tests.test_training_matrix import _prediction_rows, _write_inputs


def test_valid_lr_artifact_evaluates_plots_and_validates(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "lr",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )

    metrics = evaluate_experiment(result.output_dir, validate=True)

    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["overall"]["ranking_metrics_available"] is True
    assert metrics["ranking_metric_input"] == "prob_fake"
    for file_name in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
        assert (result.output_dir / file_name).stat().st_size > 0
    validate_experiment_artifacts(result.output_dir)


def test_validator_rejects_invalid_probability_range(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "lr_bad_prob",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    evaluate_experiment(result.output_dir)
    rows = _prediction_rows(result.predictions_path)
    rows[0]["prob_fake"] = "1.5"
    _write_prediction_rows(result.predictions_path, rows)

    with pytest.raises(ArtifactValidationError, match="prob_fake"):
        validate_experiment_artifacts(result.output_dir)


def test_validator_rejects_missing_file_empty_plot_and_row_count_mismatch(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "lr_bad_artifact",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    evaluate_experiment(result.output_dir)
    (result.output_dir / "model.joblib").unlink()
    (result.output_dir / "roc_curve.png").write_bytes(b"")
    rows = _prediction_rows(result.predictions_path)[:-1]
    _write_prediction_rows(result.predictions_path, rows)

    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_experiment_artifacts(result.output_dir)
    message = str(exc_info.value)
    assert "missing required artifact: model.joblib" in message
    assert "plot artifact is empty: roc_curve.png" in message
    assert "row count" in message


def test_score_only_svm_uses_decision_score_for_ranking_without_fake_probabilities(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "svm",
        mode="fusion",
        classifier="linear_svm",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=500,
        verify_reload=False,
    )

    metrics = evaluate_experiment(result.output_dir, validate=True)

    assert metrics["probability_supported"] is False
    assert metrics["decision_score_only"] is True
    assert metrics["ranking_metric_input"] == "decision_score"
    assert metrics["overall"]["ranking_metrics_available"] is True
    assert all(row["prob_fake"] == "" for row in _prediction_rows(result.predictions_path))
    validate_experiment_artifacts(result.output_dir)


def test_validator_rejects_empty_plot_in_copied_artifact(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "lr_source",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    evaluate_experiment(result.output_dir, validate=True)
    copied_dir = tmp_path / "lr_copied"
    shutil.copytree(result.output_dir, copied_dir)
    (copied_dir / "pr_curve.png").write_bytes(b"")

    with pytest.raises(ArtifactValidationError, match="plot artifact is empty: pr_curve.png"):
        validate_experiment_artifacts(copied_dir)


def test_one_class_split_marks_ranking_metrics_unavailable(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "one_class",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    rows = _prediction_rows(result.predictions_path)
    for row in rows:
        if row["split"] == "test":
            row["label"] = "1"
    _write_prediction_rows(result.predictions_path, rows)

    metrics = evaluate_experiment(result.output_dir, split="test")

    assert metrics["overall"]["ranking_metrics_available"] is False
    assert metrics["overall"]["roc_auc"] is None
    assert (result.output_dir / "roc_curve.png").stat().st_size > 0


def _write_prediction_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_metrics_json_contains_required_validator_keys(tmp_path: Path) -> None:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / "lr_metric_keys",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    evaluate_experiment(result.output_dir, validate=True)

    with result.metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "confusion_matrix"]:
        assert key in metrics["overall"]
