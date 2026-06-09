from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from _path import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate Phase A quick smoke and Phase B local experiment pipelines without downloading datasets."
    )
    parser.add_argument("--quick", action="store_true", help="Run a CPU-friendly synthetic REAL/FAKE smoke pipeline.")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/run_all_experiments"), help="Directory for manifests, caches, experiments, and run summary.")
    parser.add_argument("--data_root", type=Path, default=None, help="Existing local CIFAKE-style root required for full mode.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic manifest splitting and classifier seed.")
    parser.add_argument("--quick_include_clip", action="store_true", help="Also run a tiny CLIP smoke in quick mode; may require model availability/network.")
    parser.add_argument("--include_clip", action="store_true", help="Run CLIP extraction and CLIP/fusion experiments in full mode.")
    parser.add_argument("--max_samples_per_class", type=int, default=None, help="Optional per-class cap for full-mode manifest preparation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.quick:
        return _run_quick(args, output_root)
    return _run_full(args, output_root)


def _run_quick(args: argparse.Namespace, output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _new_summary(mode="quick", args=args, output_root=output_root)
    data_root = output_root / "quick_data"
    manifest_path = output_root / "manifests" / "quick_manifest.csv"
    frequency_cache = output_root / "caches" / "quick_frequency.pt"
    clip_cache = output_root / "caches" / "quick_clip.pt"
    experiment_dir = output_root / "experiments" / "quick_frequency_logistic_regression"

    try:
        _write_quick_images(data_root)
        summary["artifacts"]["quick_data_root"] = data_root.as_posix()

        steps = [
            _prepare_manifest_step(data_root, manifest_path, args.seed, num_per_class=4),
            _frequency_step(manifest_path, frequency_cache, args.seed),
            _train_step(
                manifest_path,
                experiment_dir,
                args.seed,
                mode="frequency_only",
                classifier="logistic_regression",
                frequency_cache=frequency_cache,
                clip_cache=None,
                max_iter=200,
            ),
            _evaluate_step(experiment_dir),
            _validate_step(experiment_dir),
        ]
        for step in steps:
            _run_step(step, summary)

        if args.quick_include_clip:
            blocker_path = output_root / "blockers" / "quick_clip_blocker.txt"
            clip_step = _clip_step(manifest_path, clip_cache, args.seed, max_samples=2, smoke=True, blocker_path=blocker_path)
            result = _run_step(clip_step, summary, allow_failure=True)
            if result["returncode"] != 0:
                summary["blockers"].append(
                    {
                        "step": clip_step["name"],
                        "reason": "Optional quick CLIP smoke did not complete; see command stderr and blocker evidence if written.",
                        "blocker_path": blocker_path.as_posix(),
                    }
                )
                summary["status"] = "blocked"
                _record_artifacts(summary, manifest_path, frequency_cache, experiment_dir, clip_cache if clip_cache.exists() else None)
                _write_summary(output_root / "quick_summary.json", summary)
                return int(result["returncode"] or 1)
            summary["artifacts"]["quick_clip_cache"] = clip_cache.as_posix()
        else:
            summary["skipped"].append({"step": "clip_quick_smoke", "reason": "--quick_include_clip was not requested; quick mode avoids mandatory CLIP downloads."})

        summary["status"] = "success"
        _record_artifacts(summary, manifest_path, frequency_cache, experiment_dir, clip_cache if clip_cache.exists() else None)
        _write_summary(output_root / "quick_summary.json", summary)
        print(f"quick run complete: {output_root / 'quick_summary.json'}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        _write_summary(output_root / "quick_summary.json", summary)
        print(f"quick run failed: {error}", file=sys.stderr)
        return 1


def _run_full(args: argparse.Namespace, output_root: Path) -> int:
    if args.data_root is None:
        print("full mode requires --data_root pointing to an existing local CIFAKE-style REAL/FAKE image root", file=sys.stderr)
        return 1
    data_root = args.data_root.resolve()
    if not data_root.exists() or not data_root.is_dir():
        print(f"missing data root for full mode: {data_root}", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    summary = _new_summary(mode="full", args=args, output_root=output_root)
    summary["data_root"] = data_root.as_posix()
    manifest_path = output_root / "manifests" / "full_manifest.csv"
    frequency_cache = output_root / "caches" / "full_frequency.pt"
    clip_cache = output_root / "caches" / "full_clip.pt"

    try:
        num_per_class = args.max_samples_per_class
        _run_step(_prepare_manifest_step(data_root, manifest_path, args.seed, num_per_class=num_per_class), summary)
        _run_step(_frequency_step(manifest_path, frequency_cache, args.seed), summary)

        experiments: list[tuple[str, str, Path]] = [
            ("frequency_only", "logistic_regression", output_root / "experiments" / "frequency_only_logistic_regression"),
            ("frequency_only", "linear_svm", output_root / "experiments" / "frequency_only_linear_svm"),
        ]

        if args.include_clip:
            _run_step(_clip_step(manifest_path, clip_cache, args.seed, max_samples=None, smoke=False, blocker_path=None), summary)
            experiments.extend(
                [
                    ("clip_only", "logistic_regression", output_root / "experiments" / "clip_only_logistic_regression"),
                    ("clip_only", "linear_svm", output_root / "experiments" / "clip_only_linear_svm"),
                    ("fusion", "logistic_regression", output_root / "experiments" / "fusion_logistic_regression"),
                    ("fusion", "linear_svm", output_root / "experiments" / "fusion_linear_svm"),
                ]
            )
        else:
            summary["skipped"].append({"step": "clip_full_pipeline", "reason": "--include_clip was not requested; CLIP/fusion experiments require an explicit opt-in."})

        for mode, classifier, experiment_dir in experiments:
            _run_step(
                _train_step(
                    manifest_path,
                    experiment_dir,
                    args.seed,
                    mode=mode,
                    classifier=classifier,
                    frequency_cache=frequency_cache if mode in {"frequency_only", "fusion"} else None,
                    clip_cache=clip_cache if mode in {"clip_only", "fusion"} else None,
                    max_iter=1000,
                ),
                summary,
            )
            _run_step(_evaluate_step(experiment_dir), summary)
            _run_step(_validate_step(experiment_dir), summary)

        summary["status"] = "success"
        _record_artifacts(summary, manifest_path, frequency_cache, None, clip_cache if clip_cache.exists() else None)
        _write_summary(output_root / "run_summary.json", summary)
        print(f"full run complete: {output_root / 'run_summary.json'}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        _write_summary(output_root / "run_summary.json", summary)
        print(f"full run failed: {error}", file=sys.stderr)
        return 1


def _prepare_manifest_step(data_root: Path, manifest_path: Path, seed: int, *, num_per_class: int | None) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/prepare_cifake_subset.py",
        "--data_root",
        data_root.as_posix(),
        "--output_manifest",
        manifest_path.as_posix(),
        "--seed",
        str(seed),
    ]
    if num_per_class is not None:
        command.extend(["--num_real", str(num_per_class), "--num_fake", str(num_per_class)])
    return {"name": "prepare_manifest", "command": command, "artifacts": [manifest_path.as_posix()]}


def _frequency_step(manifest_path: Path, frequency_cache: Path, seed: int) -> dict[str, Any]:
    return {
        "name": "extract_frequency_features",
        "command": [
            sys.executable,
            "scripts/extract_frequency_features.py",
            "--manifest",
            manifest_path.as_posix(),
            "--output_cache",
            frequency_cache.as_posix(),
            "--seed",
            str(seed),
        ],
        "artifacts": [frequency_cache.as_posix()],
    }


def _clip_step(manifest_path: Path, clip_cache: Path, seed: int, *, max_samples: int | None, smoke: bool, blocker_path: Path | None) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/extract_clip_features.py",
        "--manifest",
        manifest_path.as_posix(),
        "--output_cache",
        clip_cache.as_posix(),
        "--seed",
        str(seed),
        "--device",
        "cpu" if smoke else "auto",
    ]
    if max_samples is not None:
        command.extend(["--max_samples", str(max_samples)])
    if smoke:
        command.append("--smoke")
    if blocker_path is not None:
        command.extend(["--write_blocker", blocker_path.as_posix()])
    return {"name": "extract_clip_features", "command": command, "artifacts": [clip_cache.as_posix()]}


def _train_step(
    manifest_path: Path,
    experiment_dir: Path,
    seed: int,
    *,
    mode: str,
    classifier: str,
    frequency_cache: Path | None,
    clip_cache: Path | None,
    max_iter: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/train_classifier.py",
        "--manifest",
        manifest_path.as_posix(),
        "--output_dir",
        experiment_dir.as_posix(),
        "--mode",
        mode,
        "--classifier",
        classifier,
        "--seed",
        str(seed),
        "--max_iter",
        str(max_iter),
    ]
    if frequency_cache is not None:
        command.extend(["--frequency_cache", frequency_cache.as_posix()])
    if clip_cache is not None:
        command.extend(["--clip_cache", clip_cache.as_posix()])
    return {"name": f"train_{mode}_{classifier}", "command": command, "artifacts": [experiment_dir.as_posix()]}


def _evaluate_step(experiment_dir: Path) -> dict[str, Any]:
    return {
        "name": f"evaluate_{experiment_dir.name}",
        "command": [sys.executable, "scripts/evaluate.py", "--experiment_dir", experiment_dir.as_posix(), "--validate"],
        "artifacts": [(experiment_dir / "metrics.json").as_posix()],
    }


def _validate_step(experiment_dir: Path) -> dict[str, Any]:
    return {
        "name": f"validate_{experiment_dir.name}",
        "command": [sys.executable, "scripts/validate_artifacts.py", "--experiment_dir", experiment_dir.as_posix()],
        "artifacts": [(experiment_dir / "config.yaml").as_posix(), (experiment_dir / "predictions.csv").as_posix()],
    }


def _run_step(step: dict[str, Any], summary: dict[str, Any], *, allow_failure: bool = False) -> dict[str, Any]:
    command = [str(part) for part in step["command"]]
    print(f"running {step['name']}: {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    result = {
        "name": step["name"],
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "artifacts": step.get("artifacts", []),
    }
    summary["steps"].append(result)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"step {step['name']} failed with exit code {completed.returncode}")
    return result


def _write_quick_images(data_root: Path) -> None:
    for class_name, base_color in {"REAL": (32, 96, 160), "FAKE": (180, 72, 32)}.items():
        for index in range(4):
            path = data_root / class_name / f"{class_name.lower()}_{index:02d}.png"
            _write_pattern_image(path, base_color, index)


def _write_pattern_image(path: Path, base_color: tuple[int, int, int], index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), base_color)
    pixels = image.load()
    if pixels is None:
        raise RuntimeError("failed to create quick smoke image pixels")
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (
                (base_color[0] + index * 19 + x * 5 + y * 3) % 256,
                (base_color[1] + index * 17 + x * 2 + y * 7) % 256,
                (base_color[2] + index * 13 + x * 11 + y * 4) % 256,
            )
    image.save(path, format="PNG")


def _new_summary(*, mode: str, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "run_all_experiments_summary_v1",
        "mode": mode,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": output_root.as_posix(),
        "seed": int(args.seed),
        "commands_run_from": PROJECT_ROOT.as_posix(),
        "steps": [],
        "artifacts": {},
        "skipped": [],
        "blockers": [],
    }


def _record_artifacts(summary: dict[str, Any], manifest_path: Path, frequency_cache: Path, experiment_dir: Path | None, clip_cache: Path | None) -> None:
    summary["artifacts"].update(
        {
            "manifest": manifest_path.as_posix(),
            "frequency_cache": frequency_cache.as_posix(),
        }
    )
    if experiment_dir is not None:
        summary["artifacts"].update(
            {
                "experiment_dir": experiment_dir.as_posix(),
                "metrics": (experiment_dir / "metrics.json").as_posix(),
                "predictions": (experiment_dir / "predictions.csv").as_posix(),
            }
        )
    if clip_cache is not None:
        summary["artifacts"]["clip_cache"] = clip_cache.as_posix()


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
