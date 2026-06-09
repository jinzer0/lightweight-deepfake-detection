from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportAny=false, reportUnusedCallResult=false

import json
from pathlib import Path

from src.eval import evaluate_experiment
from src.train.frequency_lr import TrainResult, train_classifier
from scripts.validate_artifacts import audit_experiment, audit_robustness, write_final_report
from tests.test_eval_artifacts import _write_prediction_rows
from tests.test_training_matrix import _prediction_rows, _write_inputs


def test_final_audit_catches_row_count_mismatch(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "row_count")
    rows = _prediction_rows(result.predictions_path)[:-1]
    _write_prediction_rows(result.predictions_path, rows)

    audit = audit_experiment(result.output_dir)

    assert audit.status == "fail"
    assert any("row count" in finding.message for finding in audit.findings)


def test_final_audit_catches_invalid_probability(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "bad_probability")
    rows = _prediction_rows(result.predictions_path)
    rows[0]["prob_fake"] = "-0.01"
    _write_prediction_rows(result.predictions_path, rows)

    audit = audit_experiment(result.output_dir)

    assert audit.status == "fail"
    assert any("prob_fake" in finding.message for finding in audit.findings)


def test_final_audit_catches_non_finite_metric(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "bad_metric")
    with result.metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)
    metrics["overall"]["accuracy"] = float("nan")
    with result.metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj)

    audit = audit_experiment(result.output_dir)

    assert audit.status == "fail"
    assert any("non-finite" in finding.message or "finite" in finding.message for finding in audit.findings)


def test_final_audit_catches_empty_plot(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "empty_plot")
    (result.output_dir / "roc_curve.png").write_bytes(b"")

    audit = audit_experiment(result.output_dir)

    assert audit.status == "fail"
    assert any("empty" in finding.message and "roc_curve.png" in finding.message for finding in audit.findings)


def test_final_audit_catches_prediction_split_leakage(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "leakage")
    rows = _prediction_rows(result.predictions_path)
    train_row = next(row for row in rows if row["split"] == "train")
    test_row = next(row for row in rows if row["split"] == "test")
    test_row["path"] = train_row["path"]
    _write_prediction_rows(result.predictions_path, rows)

    audit = audit_experiment(result.output_dir)

    assert audit.status == "fail"
    assert any(finding.area == "leakage" for finding in audit.findings)


def test_final_report_marks_missing_robustness_clip_and_cuda_as_not_success(tmp_path: Path) -> None:
    result = _valid_artifact(tmp_path, "report")
    robustness = audit_robustness(tmp_path)
    report_path = write_final_report(tmp_path / "final_report.md", [audit_experiment(result.output_dir)], robustness, audit_root=tmp_path)

    report = report_path.read_text(encoding="utf-8")

    assert "## Implemented" in report
    assert "## Blocked" in report
    assert "## Deferred" in report
    assert "## Next Improvements" in report
    assert "## Deferred Items and Blockers" not in report
    assert "`src/data/`" in report
    assert "`src/features/`" in report
    assert "`src/train/`" in report
    assert "`src/eval/`" in report
    assert "`src/inference/`" in report
    assert "`app/streamlit_app.py`" in report
    assert "`README.md`" in report
    assert "not-run: robustness evidence was not discovered" in report
    assert "missing full CLIP evidence" in report
    assert "missing remote CUDA evidence" in report
    assert "not success" in report
    blocked_section = report.split("## Blocked", maxsplit=1)[1].split("## Deferred", maxsplit=1)[0]
    implemented_section = report.split("## Implemented", maxsplit=1)[1].split("## Blocked", maxsplit=1)[0]
    assert "missing full CLIP evidence" in blocked_section
    assert "missing remote CUDA evidence" in blocked_section
    assert "missing full CLIP evidence" not in implemented_section
    assert "missing remote CUDA evidence" not in implemented_section
    assert "Phase C MLP" in report


def _valid_artifact(tmp_path: Path, name: str) -> TrainResult:
    manifest_path, frequency_path, clip_path = _write_inputs(tmp_path / name)
    result = train_classifier(
        manifest_path=manifest_path,
        output_dir=tmp_path / name / "experiment",
        mode="frequency_only",
        classifier="logistic_regression",
        frequency_cache_path=frequency_path,
        clip_cache_path=clip_path,
        max_iter=200,
        verify_reload=False,
    )
    evaluate_experiment(result.output_dir, validate=True)
    return result
