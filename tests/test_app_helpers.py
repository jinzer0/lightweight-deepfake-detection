from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.app.app as app


class FakeUpload:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


class Spinner:
    def __init__(self, _message: str) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def test_save_upload_sanitizes_name_and_rejects_empty_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path)

    saved_path = app._save_upload(FakeUpload("../../messy name!!.JPEG", b"image-bytes"))

    assert saved_path.parent == tmp_path / app.UPLOAD_DIR
    assert saved_path.name.startswith("messy_name_")
    assert saved_path.suffix == ".jpeg"
    assert saved_path.read_bytes() == b"image-bytes"

    with pytest.raises(ValueError, match="Uploaded image is empty"):
        app._save_upload(FakeUpload("empty.png", b""))


@pytest.mark.parametrize(
    ("file_name", "stem", "suffix"),
    [
        ("plain.png", "plain", ".png"),
        (" spaced report final.JPG", "spaced_report_final", ".jpg"),
        ("...", "upload", ".png"),
        ("archive.gif", "archive", ".png"),
    ],
)
def test_safe_upload_name_helpers(file_name: str, stem: str, suffix: str) -> None:
    assert app._safe_stem(file_name) == stem
    assert app._safe_suffix(file_name) == suffix


def test_sync_upload_state_resets_result_when_upload_or_model_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, Any] = {"last_upload_key": None, "last_result": {"old": True}, "last_error": "old error"}
    monkeypatch.setattr(app.st, "session_state", state)

    app._sync_upload_state(FakeUpload("a.png", b"123"), "frequency_only")
    first_key = state["last_upload_key"]

    assert first_key == "a.png:3:frequency_only"
    assert state["last_result"] is None
    assert state["last_error"] is None

    state["last_result"] = {"kept": True}
    state["last_error"] = "kept"
    app._sync_upload_state(FakeUpload("a.png", b"123"), "frequency_only")
    assert state["last_upload_key"] == first_key
    assert state["last_result"] == {"kept": True}
    assert state["last_error"] == "kept"

    app._sync_upload_state(FakeUpload("a.png", b"123"), "clip_only")
    assert state["last_upload_key"] == "a.png:3:clip_only"
    assert state["last_result"] is None
    assert state["last_error"] is None


def test_run_prediction_stores_result_or_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, Any] = {}
    predicted_paths: list[Path] = []
    expected_result = {
        "ai_prob": 0.2,
        "pred_label": "Real",
        "confidence": "medium",
        "clip_score": None,
        "frequency_score": 0.2,
        "fusion_score": None,
        "spectrum_path": None,
        "radial_spectrum_path": None,
    }

    class FakeService:
        def __init__(self, config_path: Path, model_name: str) -> None:
            assert config_path == app.CONFIG_PATH
            assert model_name == "frequency_only"

        def predict(self, image_path: Path) -> dict[str, float | str | None]:
            predicted_paths.append(image_path)
            return expected_result

    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app.st, "spinner", Spinner)
    monkeypatch.setattr(app, "DetectorService", FakeService)

    app._run_prediction(FakeUpload("input.png", b"image"), "frequency_only")

    assert state["last_result"] == expected_result
    assert state["last_error"] is None
    assert predicted_paths and predicted_paths[0].is_file()

    class FailingService(FakeService):
        def predict(self, image_path: Path) -> dict[str, float | str | None]:
            raise FileNotFoundError("missing checkpoint frequency_only.pt")

    monkeypatch.setattr(app, "DetectorService", FailingService)

    app._run_prediction(FakeUpload("input.png", b"image"), "frequency_only")

    expected_error = "The selected review, Fast texture check, is not ready yet. Ask the maintainer to install the required model files."
    assert state["last_result"] is None
    assert state["last_error"] == expected_error
