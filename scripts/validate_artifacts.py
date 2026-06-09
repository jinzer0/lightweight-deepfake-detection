from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml

try:
    from _path import PROJECT_ROOT, ensure_project_root_on_path
except ModuleNotFoundError:
    from scripts._path import PROJECT_ROOT, ensure_project_root_on_path

ensure_project_root_on_path()

from src.data.manifest import read_manifest, validate_manifest_rows  # noqa: E402
from src.eval import ArtifactValidationError, validate_experiment_artifacts  # noqa: E402

PLOT_FILES = ["confusion_matrix.png", "roc_curve.png", "pr_curve.png"]
PREDICTION_COLUMNS = ["path", "sample_id", "label", "pred_label", "prob_fake", "score", "split"]
ROBUSTNESS_FILES = ["robustness_metrics.csv", "robustness_summary.png"]
REPORT_DEFAULT_PATH = PROJECT_ROOT / "outputs" / "final_implementation_report.md"


@dataclass(frozen=True)
class AuditFinding:
    status: str
    area: str
    message: str
    path: str


@dataclass(frozen=True)
class ExperimentAudit:
    path: Path
    status: str
    findings: list[AuditFinding]
    model_type: str
    feature_type: str
    streamlit_probability_eligible: bool
    probability_supported: bool
    decision_score_only: bool
    row_count: int


@dataclass(frozen=True)
class RobustnessAudit:
    path: Path | None
    status: str
    findings: list[AuditFinding]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate experiment artifacts and optionally write a final audit report.")
    parser.add_argument("--experiment_dir", type=Path, default=None, help="Experiment artifact directory to validate. Existing single-experiment behavior is preserved.")
    parser.add_argument("--audit_root", type=Path, default=None, help="Root directory to scan for generated experiment and robustness artifacts.")
    parser.add_argument("--report", action="store_true", help="Write a concise final implementation report from the audit results.")
    parser.add_argument("--output_path", type=Path, default=REPORT_DEFAULT_PATH, help="Report output path when --report is used. Defaults to outputs/final_implementation_report.md.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.experiment_dir is None and args.audit_root is None:
        print("artifact validation failed: pass --experiment_dir or --audit_root", file=sys.stderr)
        return 1

    try:
        experiments: list[ExperimentAudit] = []
        robustness: RobustnessAudit | None = None
        if args.experiment_dir is not None:
            validate_experiment_artifacts(args.experiment_dir)
            experiment_audit = audit_experiment(args.experiment_dir)
            experiments.append(experiment_audit)
            _raise_on_failed_experiment(experiment_audit)
            print(f"artifact validation passed: {args.experiment_dir}")
        if args.audit_root is not None:
            experiments.extend(audit_experiment(path) for path in discover_experiment_dirs(args.audit_root))
            robustness = audit_robustness(args.audit_root)
            for experiment in experiments:
                _raise_on_failed_experiment(experiment)
            _raise_on_failed_robustness(robustness)
            print(f"artifact audit passed: {args.audit_root} ({len(experiments)} experiment artifact directories)")
        if args.report:
            report_path = write_final_report(args.output_path, experiments, robustness, audit_root=args.audit_root)
            print(f"wrote final implementation report: {report_path}")
        return 0
    except (ArtifactValidationError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"artifact validation failed: {error}", file=sys.stderr)
        return 1


def discover_experiment_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"audit root does not exist: {root_path}")
    candidates = [path.parent for path in root_path.rglob("config.yaml")]
    return sorted({path for path in candidates if (path / "predictions.csv").exists() and (path / "metrics.json").exists()})


def audit_experiment(experiment_dir: str | Path) -> ExperimentAudit:
    directory = Path(experiment_dir)
    findings: list[AuditFinding] = []
    try:
        validate_experiment_artifacts(directory)
    except ArtifactValidationError as error:
        findings.extend(AuditFinding("fail", "artifact_schema", message, directory.as_posix()) for message in error.errors)

    config = _read_yaml(directory / "config.yaml")
    metrics = _read_json(directory / "metrics.json")
    rows = _read_predictions(directory / "predictions.csv")

    findings.extend(_validate_label_orientation(config, directory))
    findings.extend(_validate_probability_semantics(config, rows, directory))
    findings.extend(_validate_streamlit_eligibility(config, directory))
    findings.extend(_validate_manifest_alignment(config, rows, directory))
    findings.extend(_validate_prediction_leakage(rows, directory))
    findings.extend(_validate_finite_metrics(metrics, directory))
    findings.extend(_validate_plots(directory))

    status = "fail" if any(finding.status == "fail" for finding in findings) else "pass"
    classifier = _dict_value(config, "classifier")
    feature = _dict_value(config, "feature")
    return ExperimentAudit(
        path=directory,
        status=status,
        findings=findings,
        model_type=str(classifier.get("key") or classifier.get("type") or "unknown"),
        feature_type=str(config.get("mode") or feature.get("mode") or "unknown"),
        streamlit_probability_eligible=bool(config.get("streamlit_probability_eligible", False)),
        probability_supported=bool(config.get("probability_supported", False)),
        decision_score_only=bool(config.get("decision_score_only", False)),
        row_count=len(rows),
    )


def audit_robustness(root: str | Path) -> RobustnessAudit:
    root_path = Path(root)
    metrics_paths = sorted(root_path.rglob("robustness_metrics.csv"))
    findings: list[AuditFinding] = []
    if not metrics_paths:
        return RobustnessAudit(None, "not-run", [AuditFinding("not-run", "robustness", "robustness evidence was not found; status is not-run, not success", root_path.as_posix())])

    for metrics_path in metrics_paths:
        summary_path = metrics_path.with_name("robustness_summary.png")
        if not summary_path.exists():
            findings.append(AuditFinding("fail", "robustness", "missing robustness_summary.png", metrics_path.parent.as_posix()))
        elif summary_path.stat().st_size == 0:
            findings.append(AuditFinding("fail", "robustness", "robustness_summary.png is empty", summary_path.as_posix()))
        rows = _read_csv(metrics_path)
        if not rows:
            findings.append(AuditFinding("fail", "robustness", "robustness_metrics.csv has no rows", metrics_path.as_posix()))
        for index, row in enumerate(rows, start=2):
            for key in ["sample_count", "clean_accuracy", "corrupted_accuracy", "accuracy_degradation"]:
                if key in row and not _is_finite_number(row[key]):
                    findings.append(AuditFinding("fail", "robustness", f"row {index} {key} must be finite", metrics_path.as_posix()))
    status = "fail" if any(finding.status == "fail" for finding in findings) else "pass"
    return RobustnessAudit(metrics_paths[0].parent, status, findings)


def write_final_report(output_path: str | Path, experiments: list[ExperimentAudit], robustness: RobustnessAudit | None, *, audit_root: Path | None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_types = sorted({experiment.model_type for experiment in experiments if experiment.model_type != "unknown"})
    feature_types = sorted({experiment.feature_type for experiment in experiments if experiment.feature_type != "unknown"})
    streamlit_eligible = [experiment.path.as_posix() for experiment in experiments if experiment.streamlit_probability_eligible]
    blocked_items = _blocked_items(audit_root, robustness)
    deferred_items = _deferred_items()
    implemented_items = _implemented_items(model_types, feature_types, streamlit_eligible, robustness)
    lines = [
        "# Final Implementation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Created/Modified Files",
        "- Project configuration and dependency files: `README.md`, `requirements.txt`, `configs/`.",
        "- Data and manifest pipeline: `scripts/prepare_cifake_subset.py`, `src/data/`, and related tests.",
        "- Feature extraction and cache pipeline: `scripts/extract_frequency_features.py`, `scripts/extract_clip_features.py`, `src/features/`, and related tests.",
        "- Training and model artifacts pipeline: `scripts/train_classifier.py`, `src/train/`, `src/models/`, and related tests.",
        "- Evaluation, robustness, and artifact audit pipeline: `scripts/evaluate.py`, `scripts/run_robustness.py`, `scripts/validate_artifacts.py`, `src/eval/`, and related tests.",
        "- Experiment orchestration: `scripts/run_all_experiments.py`.",
        "- Inference and demo surface: `src/inference/`, `app/streamlit_app.py`, and related tests.",
        "",
        "## Runnable Commands",
        "- `python scripts/run_all_experiments.py --quick`",
        "- `python scripts/validate_artifacts.py --experiment_dir <experiment_dir>`",
        "- `python scripts/validate_artifacts.py --audit_root outputs --report`",
        "- `python -m compileall scripts src tests`",
        "- `pytest -q`",
        "",
        "## Implemented",
        _bullet_or_not_run(implemented_items),
        "",
        "## Blocked",
        _bullet_or_not_run(blocked_items),
        "",
        "## Deferred",
        _bullet_or_not_run(deferred_items),
        "",
        "## Result Structure",
        "- Experiment directories contain `config.yaml`, `metrics.json`, `predictions.csv`, model/scaler joblib files, and confusion/ROC/PR plots.",
        "- Robustness outputs, when run, contain `robustness_metrics.csv` and `robustness_summary.png`.",
        "- Final audit reports are written under `outputs/` unless `--output_path` is provided.",
        "",
        "## Audit Findings",
    ]
    if experiments:
        for experiment in experiments:
            lines.append(f"- `{experiment.path.as_posix()}`: {experiment.status}, rows={experiment.row_count}, model={experiment.model_type}, feature={experiment.feature_type}")
            for finding in experiment.findings:
                if finding.status != "pass":
                    lines.append(f"- {finding.status}: {finding.area}: {finding.message} ({finding.path})")
    else:
        lines.append("- not-run: no experiment artifact directories were discovered for this report.")
    if robustness is not None:
        lines.append(f"- robustness: {robustness.status}")
        for finding in robustness.findings:
            lines.append(f"- {finding.status}: {finding.area}: {finding.message} ({finding.path})")
    lines.extend([
        "",
        "## Next Improvements",
        "- Collect real full CIFAKE, CLIP, and remote CUDA evidence before claiming those runs as complete.",
        "- Add calibrated probability support before making score-only models Streamlit probability eligible.",
        "- Expand robustness beyond quick frequency-only checks when compute and data are available.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _implemented_items(
    model_types: list[str], feature_types: list[str], streamlit_eligible: list[str], robustness: RobustnessAudit | None
) -> list[str]:
    items = [
        "local CIFAKE-style manifest preparation and validation",
        "frequency FFT/DCT feature extraction with cache validation",
        "openCLIP feature extraction path when model/cache access is available",
        "classifier training, reload checks, metrics, predictions, and plot artifacts",
        "artifact-backed Streamlit JPG/PNG demo for probability-eligible artifacts",
        "semantic artifact audit and final report generation",
    ]
    if model_types:
        items.append(f"model types observed in audited artifacts: {', '.join(model_types)}")
    if feature_types:
        items.append(f"feature types observed in audited artifacts: {', '.join(feature_types)}")
    if streamlit_eligible:
        items.append(f"Streamlit probability-eligible artifacts: {', '.join(streamlit_eligible)}")
    if robustness is not None and robustness.status == "pass":
        items.append("robustness evidence discovered and validated")
    return items


def _deferred_items() -> list[str]:
    return [
        "Phase C MLP decision and implementation remain deferred",
        "HF loader schema confirmation remains deferred",
        "RBF/UMAP/advanced peak features remain deferred",
        "GenImage full download and external dataset claims remain deferred",
        "polished Streamlit selector enhancements remain deferred",
    ]


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _validate_label_orientation(config: dict[str, Any], directory: Path) -> list[AuditFinding]:
    mapping = config.get("label_mapping")
    if mapping != {"real": 0, "fake": 1}:
        return [AuditFinding("fail", "label_orientation", "label mapping must stay real=0 and fake=1", (directory / "config.yaml").as_posix())]
    return []


def _validate_probability_semantics(config: dict[str, Any], rows: list[dict[str, str]], directory: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    classifier = _dict_value(config, "classifier")
    classifier_key = str(classifier.get("key", ""))
    probability_supported = bool(config.get("probability_supported", False))
    decision_score_only = bool(config.get("decision_score_only", False))
    if classifier_key == "logistic_regression" and not probability_supported:
        findings.append(AuditFinding("fail", "probability", "LogisticRegression artifacts must declare probability_supported=true", (directory / "config.yaml").as_posix()))
    if probability_supported:
        for index, row in enumerate(rows, start=2):
            if not _is_probability(row.get("prob_fake", "")):
                findings.append(AuditFinding("fail", "probability", f"row {index} prob_fake must be finite in [0, 1]", (directory / "predictions.csv").as_posix()))
    if decision_score_only:
        for index, row in enumerate(rows, start=2):
            if row.get("prob_fake", "") != "":
                findings.append(AuditFinding("fail", "probability", f"row {index} score-only artifact must not fake prob_fake", (directory / "predictions.csv").as_posix()))
    return findings


def _validate_streamlit_eligibility(config: dict[str, Any], directory: Path) -> list[AuditFinding]:
    mode = str(config.get("mode", ""))
    probability_supported = bool(config.get("probability_supported", False))
    eligible = bool(config.get("streamlit_probability_eligible", False))
    classifier = _dict_value(config, "classifier")
    classifier_key = str(classifier.get("key", ""))
    expected = mode == "frequency_only" and classifier_key == "logistic_regression" and probability_supported
    if eligible != expected:
        return [AuditFinding("fail", "streamlit", "Streamlit probability eligibility must be true only for frequency_only LogisticRegression probability artifacts", (directory / "config.yaml").as_posix())]
    return []


def _validate_manifest_alignment(config: dict[str, Any], rows: list[dict[str, str]], directory: Path) -> list[AuditFinding]:
    inputs = _dict_value(config, "inputs")
    manifest_value = inputs.get("manifest_path")
    if not manifest_value:
        return [AuditFinding("warning", "manifest", "config has no manifest_path; split row counts and manifest leakage could not be checked", (directory / "config.yaml").as_posix())]
    manifest_path = Path(str(manifest_value))
    if not manifest_path.exists():
        return [AuditFinding("warning", "manifest", f"manifest_path is missing; split row counts could not be checked: {manifest_path}", (directory / "config.yaml").as_posix())]
    manifest_rows = read_manifest(manifest_path)
    manifest_errors = validate_manifest_rows(manifest_rows, strict=False)
    findings = [AuditFinding("fail", "manifest", error, manifest_path.as_posix()) for error in manifest_errors if "leakage" in error or "duplicate" in error]
    manifest_counts = Counter(row.get("split", "") for row in manifest_rows)
    prediction_counts = Counter(row.get("split", "") for row in rows)
    for split_name, expected_count in sorted(manifest_counts.items()):
        actual_count = prediction_counts.get(split_name, 0)
        if actual_count != expected_count:
            findings.append(AuditFinding("fail", "row_count", f"split {split_name!r} row count {actual_count} does not match manifest split count {expected_count}", (directory / "predictions.csv").as_posix()))
    return findings


def _validate_prediction_leakage(rows: list[dict[str, str]], directory: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for key in ["sample_id", "path"]:
        split_map: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(key, "")).strip()
            split = str(row.get("split", "")).strip()
            if value and split:
                split_map[value].add(split)
        for value, splits in sorted(split_map.items()):
            if "train" in splits and ("test" in splits or "val" in splits):
                findings.append(AuditFinding("fail", "leakage", f"{key} {value!r} appears across splits {sorted(splits)}", (directory / "predictions.csv").as_posix()))
    return findings


def _validate_finite_metrics(metrics: Any, directory: Path) -> list[AuditFinding]:
    bad_paths: list[str] = []

    def walk(value: Any, name: str) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, f"{name}.{child_key}" if name else str(child_key))
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                walk(child_value, f"{name}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            bad_paths.append(name)
        elif isinstance(value, str) and value.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}:
            bad_paths.append(name)

    walk(metrics, "")
    return [AuditFinding("fail", "metrics", f"metrics.json contains non-finite value at {bad_path}", (directory / "metrics.json").as_posix()) for bad_path in bad_paths]


def _validate_plots(directory: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for file_name in PLOT_FILES:
        path = directory / file_name
        if not path.exists():
            findings.append(AuditFinding("fail", "plot", f"missing plot artifact: {file_name}", path.as_posix()))
        elif path.stat().st_size == 0:
            findings.append(AuditFinding("fail", "plot", f"plot artifact is empty: {file_name}", path.as_posix()))
    return findings


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError(f"predictions.csv columns must be exactly {PREDICTION_COLUMNS}, got {reader.fieldnames}")
        return list(reader)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _is_probability(value: Any) -> bool:
    if not _is_finite_number(value):
        return False
    number = float(value)
    return 0.0 <= number <= 1.0


def _raise_on_failed_experiment(experiment: ExperimentAudit) -> None:
    failures = [finding.message for finding in experiment.findings if finding.status == "fail"]
    if failures:
        raise ArtifactValidationError(failures)


def _raise_on_failed_robustness(robustness: RobustnessAudit | None) -> None:
    if robustness is None:
        return
    failures = [finding.message for finding in robustness.findings if finding.status == "fail"]
    if failures:
        raise ArtifactValidationError(failures)


def _blocked_items(audit_root: Path | None, robustness: RobustnessAudit | None) -> list[str]:
    items = [
        "not-run: missing full CLIP evidence unless CLIP experiment directories and logs are present",
        "blocked/not-run: missing remote CUDA evidence unless separate run logs record CUDA device, command, and outputs",
    ]
    if robustness is None or robustness.status == "not-run":
        items.append("not-run: robustness evidence was not discovered in the audited outputs")
    if audit_root is not None:
        blocker_files = sorted(Path(audit_root).rglob("*blocker*"))
        items.extend(f"blocked: evidence file `{path.as_posix()}`" for path in blocker_files)
    return items


def _bullet_or_not_run(values: list[str]) -> str:
    if not values:
        return "- not-run: no evidence discovered."
    return "\n".join(f"- {value}" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
