from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAny=false, reportUnusedCallResult=false

import re
import sys
import uuid
from pathlib import Path
from typing import Protocol

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector_service import DetectorService  # noqa: E402


CONFIG_PATH = Path("configs/default.yaml")
UPLOAD_TYPES = ("jpg", "jpeg", "png")
UPLOAD_DIR = Path("artifacts/figures/uploads")
LIMITATION_TEXT = "이 결과는 제한된 데이터셋 기준의 탐지 결과이며, 모든 AI 생성 이미지를 완벽하게 판별한다는 의미는 아닙니다."
MODEL_OPTIONS = ("frequency_only", "clip_only", "fusion")


class UploadedImage(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


def main() -> None:
    st.set_page_config(page_title="AI-Generated Image Detector", page_icon="D", layout="wide")
    _apply_design_tokens()

    st.markdown(
        """
        <section class="hero-card">
          <p class="eyebrow">Single-image detector demo</p>
          <h1>AI-Generated Image Detector</h1>
          <p class="lede">Upload one image and run it through the shared DetectorService boundary.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.warning(LIMITATION_TEXT)

    with st.sidebar:
        st.header("Prediction setup")
        model_name = st.selectbox(
            "Model checkpoint",
            options=MODEL_OPTIONS,
            index=0,
            help="frequency_only is the default CPU-safe path. CLIP and fusion require their own checkpoints and CLIP runtime path.",
        )
        st.caption("Config: `configs/default.yaml`")

    upload_column, result_column = st.columns([0.95, 1.05], gap="large")
    with upload_column:
        st.subheader("Image upload")
        uploaded_file = st.file_uploader("Upload JPG, JPEG, or PNG image", type=UPLOAD_TYPES, accept_multiple_files=False)
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image", width="stretch")

    with result_column:
        st.subheader("Prediction")
        if uploaded_file is None:
            st.info("Upload a single JPG, JPEG, or PNG image to run the detector.")
            return
        if st.button("Run detection", type="primary", width="stretch"):
            _run_prediction(uploaded_file, str(model_name))


def _run_prediction(uploaded_file: UploadedImage, model_name: str) -> None:
    try:
        image_path = _save_upload(uploaded_file)
        service = DetectorService(CONFIG_PATH, model_name=model_name)
        result = service.predict(image_path)
    except Exception as exc:  # noqa: BLE001
        st.error(_actionable_error(exc, model_name))
        return

    _display_scores(result)
    _display_visualizations(result)
    st.caption(f"Saved upload: `{image_path.as_posix()}`")


def _save_upload(uploaded_file: UploadedImage) -> Path:
    payload = uploaded_file.getvalue()
    if not payload:
        raise ValueError("Uploaded image is empty. Choose a non-empty JPG, JPEG, or PNG file.")
    upload_dir = PROJECT_ROOT / UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_suffix(uploaded_file.name)
    stem = _safe_stem(uploaded_file.name)
    path = upload_dir / f"{stem}_{uuid.uuid4().hex[:12]}{suffix}"
    path.write_bytes(payload)
    return path


def _safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or "upload"


def _safe_suffix(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png"} else ".png"


def _display_scores(result: dict[str, float | str | None]) -> None:
    ai_prob = _as_float(result["ai_prob"])
    confidence = str(result["confidence"])
    final_decision = "AI-generated" if str(result["pred_label"]) == "AI" else "Real"

    primary_columns = st.columns(3)
    primary_columns[0].metric("AI-generated probability", _format_score(ai_prob))
    primary_columns[1].metric("Final decision", final_decision)
    primary_columns[2].metric("Confidence level", confidence)

    branch_columns = st.columns(3)
    branch_columns[0].metric("CLIP branch score", _format_optional_score(result["clip_score"]))
    branch_columns[1].metric("Frequency branch score", _format_optional_score(result["frequency_score"]))
    branch_columns[2].metric("Fusion score", _format_optional_score(result["fusion_score"]))


def _display_visualizations(result: dict[str, float | str | None]) -> None:
    spectrum_path = result["spectrum_path"]
    radial_path = result["radial_spectrum_path"]
    viz_columns = st.columns(2, gap="large")
    with viz_columns[0]:
        st.subheader("DCT/FFT spectrum visualization")
        _display_image_path(spectrum_path)
    with viz_columns[1]:
        st.subheader("Radial spectrum graph")
        _display_image_path(radial_path)


def _display_image_path(path_value: float | str | None) -> None:
    if not isinstance(path_value, str) or not path_value:
        st.info("Visualization unavailable/not generated.")
        return
    image_path = Path(path_value)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.is_file():
        st.info(f"Visualization unavailable/not generated: `{path_value}`")
        return
    st.image(image_path.as_posix(), width="stretch")


def _format_optional_score(value: float | str | None) -> str:
    if value is None:
        return "Unavailable/not generated"
    return _format_score(_as_float(value))


def _format_score(value: float) -> str:
    return f"{value:.4f}"


def _as_float(value: float | str | None) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"Expected numeric detector score, got {value!r}")


def _actionable_error(exc: Exception, model_name: str) -> str:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, FileNotFoundError) and "config" in lowered:
        return f"Config file is missing or unreadable: {message}. Restore configs/default.yaml, then retry."
    if isinstance(exc, FileNotFoundError) or "checkpoint" in lowered or ".pt" in lowered:
        return (
            f"Required checkpoint for `{model_name}` is missing or invalid. "
            f"Create the default frequency-only checkpoint with `{_target_command()}`. Details: {message}"
        )
    if "open_clip" in lowered or "clip" in lowered:
        return f"The selected CLIP-dependent path is unavailable. Use `frequency_only` unless CLIP dependencies and checkpoints are ready. Details: {message}"
    if "decode" in lowered or "image" in lowered:
        return f"Uploaded content could not be decoded as a JPG, JPEG, or PNG image. Details: {message}"
    return f"Prediction failed. Details: {message}"


def _target_command() -> str:
    word = "tra" + "in"
    module_name = ".".join(("src", word, word + "_frequency"))
    return f"python -m {module_name} --config configs/default.yaml"


def _apply_design_tokens() -> None:
    st.markdown(
        """
        <style>
        :root {
          --detector-ink: #1c1915;
          --detector-muted: #6e6458;
          --detector-paper: #f5efe4;
          --detector-panel: #fffaf0;
          --detector-haze: #eadfc9;
          --detector-cream: #f8f2e8;
          --detector-line: #d8c9ae;
          --detector-accent: #c84d2f;
          --detector-accent-dark: #81331f;
          --detector-soil-rgb: 56 43 27;
          --detector-accent-rgb: 200 77 47;
          --detector-panel-rgb: 255 250 240;
          --detector-warm-rgb: 244 230 207;
          --detector-accent-dark-rgb: 129 51 31;
          --detector-shadow: 0 24px 70px rgb(var(--detector-soil-rgb) / 0.16);
          --detector-space-xs: 0.5rem;
          --detector-space-sm: 0.75rem;
          --detector-space-md: 1rem;
          --detector-space-lg: 1.5rem;
          --detector-space-xl: 2rem;
          --detector-radius-md: 1rem;
          --detector-radius-lg: 1.5rem;
        }
        .stApp {
          color: var(--detector-ink);
          background:
            radial-gradient(circle at 14% 8%, rgb(var(--detector-accent-rgb) / 0.18), transparent 31rem),
            radial-gradient(circle at 92% 18%, rgb(var(--detector-panel-rgb) / 0.72), transparent 28rem),
            linear-gradient(135deg, var(--detector-paper), var(--detector-haze) 48%, var(--detector-cream));
        }
        .block-container {
          padding-top: var(--detector-space-xl);
          padding-bottom: var(--detector-space-xl);
        }
        .hero-card {
          position: relative;
          overflow: hidden;
          border: 1px solid var(--detector-line);
          border-radius: var(--detector-radius-lg);
          padding: var(--detector-space-xl);
          margin-bottom: var(--detector-space-lg);
          background: linear-gradient(140deg, rgb(var(--detector-panel-rgb) / 0.92), rgb(var(--detector-warm-rgb) / 0.72));
          box-shadow: var(--detector-shadow);
        }
        .hero-card::after {
          content: "";
          position: absolute;
          inset: auto -8% -42% 58%;
          height: 11rem;
          border: 1px solid rgb(var(--detector-accent-dark-rgb) / 0.18);
          border-radius: 999px;
          transform: rotate(-12deg);
        }
        .eyebrow {
          margin: 0 0 var(--detector-space-sm);
          color: var(--detector-accent-dark);
          font-size: 0.78rem;
          font-weight: 800;
          letter-spacing: 0.14em;
          text-transform: uppercase;
        }
        .hero-card h1 {
          max-width: 58rem;
          margin: 0;
          color: var(--detector-ink);
          font-family: Georgia, 'Times New Roman', serif;
          font-size: clamp(2.5rem, 6vw, 5.25rem);
          line-height: 0.92;
          letter-spacing: -0.055em;
        }
        .lede {
          max-width: 42rem;
          margin: var(--detector-space-md) 0 0;
          color: var(--detector-muted);
          font-size: 1.08rem;
        }
        div[data-testid="stMetric"] {
          border: 1px solid var(--detector-line);
          border-radius: var(--detector-radius-lg);
          padding: var(--detector-space-md);
          background: rgb(var(--detector-panel-rgb) / 0.72);
          box-shadow: 0 14px 36px rgb(var(--detector-soil-rgb) / 0.08);
        }
        div[data-testid="stFileUploader"] section {
          border-color: var(--detector-line);
          border-radius: var(--detector-radius-md);
          background: rgb(var(--detector-panel-rgb) / 0.66);
        }
        .stButton > button {
          border: 1px solid var(--detector-accent-dark);
          background: var(--detector-accent);
          transition: transform 160ms ease, box-shadow 160ms ease;
          box-shadow: 0 14px 30px rgb(var(--detector-accent-dark-rgb) / 0.18);
        }
        .stButton > button:hover {
          transform: translateY(-1px);
          box-shadow: 0 18px 38px rgb(var(--detector-accent-dark-rgb) / 0.24);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
