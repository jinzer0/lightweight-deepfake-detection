from __future__ import annotations

from pathlib import Path

import pytest

import src.eval.robustness_runner as robustness_runner
from src.eval.robustness_runner import resolve_checkpoint_path


def test_resolve_checkpoint_path_finds_existing_candidate(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "fusion.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_checkpoint_path("fusion", {}, checkpoint_dir)

    assert resolved == checkpoint


def test_resolve_checkpoint_path_reports_tried_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = tmp_path / "missing_checkpoints"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(robustness_runner, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_checkpoint_path("frequency_only", {"paths": {"checkpoint_dir": str(checkpoint_dir)}}, checkpoint_dir)

    message = str(exc_info.value)
    assert "No checkpoint found for frequency_only" in message
    assert "frequency_only.pt" in message
    assert checkpoint_dir.as_posix() in message
