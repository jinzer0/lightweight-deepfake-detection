from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedCallResult=false

import re
import sys
import uuid
from html import escape
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
LIMITATION_TEXT = "Research demo · not forensic proof"
MODEL_OPTIONS = ("frequency_only", "clip_only", "fusion")
MODEL_LABELS = {
    "frequency_only": "Fast texture check",
    "clip_only": "Visual similarity check",
    "fusion": "Balanced combined check",
}
MODEL_DESCRIPTIONS = {
    "frequency_only": "Best for a quick local demo. Uses radial frequency texture cues.",
    "clip_only": "Uses broader visual cues. May need CLIP weights available locally.",
    "fusion": "Combines visual and texture signals when all model files are installed.",
}


class UploadedImage(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


def main() -> None:
    st.set_page_config(page_title="AI-Generated Image Detector", page_icon="D", layout="wide")
    _apply_design_tokens()
    _init_state()

    _display_hero()

    model_name, uploaded_file, analyze_clicked = _display_upload_panel()
    _sync_upload_state(uploaded_file, str(model_name))

    if analyze_clicked and uploaded_file is not None:
        _run_prediction(uploaded_file, str(model_name))

    _display_report_area(uploaded_file)


def _init_state() -> None:
    st.session_state.setdefault("last_upload_key", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)


def _display_hero() -> None:
    st.markdown(
        f"""
        <section class="hero-card">
          <p class="eyebrow">AI image detection demo</p>
          <h1>AI-GEN Detector</h1>
          <p class="lede">Upload one image and generate a reference detector report.</p>
          <p class="limitation">{escape(LIMITATION_TEXT)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _display_upload_panel() -> tuple[str, UploadedImage | None, bool]:
    with st.container(border=True):
        upload_column, preview_column = st.columns([0.42, 0.58], gap="large", vertical_alignment="center")

        with upload_column:
            st.markdown(
                """
                <section class="panel-heading">
                  <p class="eyebrow">Upload</p>
                  <h2>Analyze one image</h2>
                </section>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Drag and drop an image here",
                type=UPLOAD_TYPES,
                accept_multiple_files=False,
                help="JPG, JPEG, or PNG. Keep files small for a smooth demo.",
                label_visibility="collapsed",
            )
            model_name = st.selectbox(
                "Model",
                options=MODEL_OPTIONS,
                index=0,
                format_func=_model_label,
                help="Use Fast texture check for the quickest demo path.",
            )
            st.caption(MODEL_DESCRIPTIONS[str(model_name)])
            analyze_clicked = st.button(
                "Analyze image",
                type="primary",
                width="stretch",
                disabled=uploaded_file is None,
            )

        with preview_column:
            if uploaded_file is None:
                st.markdown(
                    """
                    <section class="preview-empty">
                      <strong>Preview</strong>
                      <p>Your selected image will appear here before analysis.</p>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.image(uploaded_file, caption="Preview", width="stretch")

    return str(model_name), uploaded_file, bool(analyze_clicked)


def _sync_upload_state(uploaded_file: UploadedImage | None, model_name: str) -> None:
    if uploaded_file is None:
        current_key = None
    else:
        payload = uploaded_file.getvalue()
        current_key = f"{uploaded_file.name}:{len(payload)}:{model_name}"
    if current_key != st.session_state.get("last_upload_key"):
        st.session_state["last_upload_key"] = current_key
        st.session_state["last_result"] = None
        st.session_state["last_error"] = None


def _display_report_area(uploaded_file: UploadedImage | None) -> None:
    result = st.session_state.get("last_result")
    error = st.session_state.get("last_error")

    if error:
        st.error(str(error))
        return
    if result is not None:
        _display_scores(result)
        _display_visualizations(result)
        return

    st.markdown(
        """
        <section class="report-empty">
          <p class="eyebrow">Report</p>
          <h2>Waiting for an image</h2>
          <p>Upload one JPG or PNG, choose a model, then run analysis.</p>
        </section>
        """
        if uploaded_file is None
        else """
        <section class="report-empty report-empty--ready">
          <p class="eyebrow">Report</p>
          <h2>Image loaded</h2>
          <p>Click <strong>Analyze image</strong> to generate the report.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _model_label(model_name: str) -> str:
    return MODEL_LABELS[model_name]


def _run_prediction(uploaded_file: UploadedImage, model_name: str) -> None:
    st.session_state["last_result"] = None
    st.session_state["last_error"] = None
    try:
        with st.spinner("Analyzing image…"):
            image_path = _save_upload(uploaded_file)
            service = DetectorService(CONFIG_PATH, model_name=model_name)
            result = service.predict(image_path)
    except Exception as exc:  # noqa: BLE001
        st.session_state["last_error"] = _actionable_error(exc, model_name)
        return

    st.session_state["last_result"] = result


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
    result_class = "result-summary--ai" if final_decision == "AI-generated" else "result-summary--real"

    st.markdown(
        f"""
        <section class="result-summary {result_class}">
          <p class="eyebrow">Result summary</p>
          <h2>{escape(final_decision)}</h2>
          <p>AI-generated likelihood {escape(_format_score(ai_prob))} · {escape(confidence)} confidence</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    primary_columns = st.columns(3)
    primary_columns[0].metric("AI-generated likelihood", _format_score(ai_prob))
    primary_columns[1].metric("Final decision", final_decision)
    primary_columns[2].metric("Confidence level", confidence)

    st.markdown('<p class="metric-group-label">Supporting signals</p>', unsafe_allow_html=True)
    branch_columns = st.columns(3)
    branch_columns[0].metric("Texture signal", _format_optional_score(result["frequency_score"]))
    branch_columns[1].metric("Visual signal", _format_optional_score(result["clip_score"]))
    branch_columns[2].metric("Combined signal", _format_optional_score(result["fusion_score"]))


def _display_visualizations(result: dict[str, float | str | None]) -> None:
    spectrum_path = result["spectrum_path"]
    radial_path = result["radial_spectrum_path"]
    st.markdown(
        """
        <section class="section-heading section-heading--compact">
          <p class="eyebrow">Optional visual evidence</p>
          <h2>Texture views</h2>
          <p>These views can help explain texture patterns when the detector produces them.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    viz_columns = st.columns(2, gap="large")
    with viz_columns[0]:
        st.subheader("Spectrum")
        _display_image_path(spectrum_path)
    with viz_columns[1]:
        st.subheader("Radial spectrum")
        _display_image_path(radial_path)


def _display_image_path(path_value: float | str | None) -> None:
    if not isinstance(path_value, str) or not path_value:
        st.info("This view is not available for the selected model.")
        return
    image_path = Path(path_value)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.is_file():
        st.info("This view is not available for the selected model.")
        return
    st.image(image_path.as_posix(), width="stretch")


def _format_optional_score(value: float | str | None) -> str:
    if value is None:
        return "—"
    return _format_score(_as_float(value))


def _format_score(value: float) -> str:
    return f"{value:.2f}"


def _as_float(value: float | str | None) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"Expected numeric detector score, got {value!r}")


def _actionable_error(exc: Exception, model_name: str) -> str:
    lowered = str(exc).lower()
    selected_review = _model_label(model_name)
    if isinstance(exc, FileNotFoundError) and "config" in lowered:
        return "This app is missing a required setup file. Ask the maintainer to check the app setup."
    if isinstance(exc, FileNotFoundError) or "checkpoint" in lowered or ".pt" in lowered:
        return (
            f"The selected review, {selected_review}, is not ready yet. "
            "Ask the maintainer to install the required model files."
        )
    if "open_clip" in lowered or "clip" in lowered:
        return f"The selected review, {selected_review}, is unavailable in this app setup. Try Fast texture check."
    if "decode" in lowered or "image" in lowered:
        return "Uploaded content could not be decoded as a JPG, JPEG, or PNG image."
    return "Prediction failed. Try another image or ask the maintainer to check the app setup."


def _apply_design_tokens() -> None:
    st.markdown(
        """
        <style>
        :root {
          --detector-bg: #f7f5f0;
          --detector-surface: #fffdf8;
          --detector-surface-muted: #f1eee6;
          --detector-ink: #20201d;
          --detector-muted: #5c574f;
          --detector-line: #ded8cc;
          --detector-accent: #344a40;
          --detector-accent-ink: #21332b;
          --detector-button-bg: #2f463b;
          --detector-button-text: #fffaf0;
          --detector-button-disabled-bg: #ebe6dc;
          --detector-button-disabled-text: #5c574f;
          --detector-danger: #7a3c2f;
          --detector-danger-soft: #f1ded8;
          --detector-success: #35543b;
          --detector-success-soft: #dfe9df;
          --detector-radius: 1rem;
          --detector-font: Aptos, IBM Plex Sans, Segoe UI, sans-serif;
          --detector-display: Iowan Old Style, Charter, Georgia, serif;
        }
        .stApp {
          color: var(--detector-ink);
          background: var(--detector-bg);
          font-family: var(--detector-font);
        }
        .block-container {
          max-width: 76rem;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3, p, label, span, div[data-testid="stMarkdownContainer"] {
          font-family: var(--detector-font);
        }
        .hero-card,
        .report-empty,
        .result-summary,
        div[data-testid="stMetric"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border: 1px solid var(--detector-line);
          border-radius: var(--detector-radius);
          background: var(--detector-surface);
        }
        .hero-card {
          margin-bottom: 1.25rem;
          padding: 2rem;
        }
        .hero-card h1 {
          margin: 0.2rem 0 0;
          color: var(--detector-ink);
          font-family: var(--detector-display);
          font-size: clamp(2.4rem, 5vw, 4.2rem);
          line-height: 0.98;
          letter-spacing: -0.045em;
        }
        .eyebrow, .metric-group-label {
          margin: 0;
          color: var(--detector-accent-ink);
          font-size: 0.78rem;
          font-weight: 750;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
        .lede {
          max-width: 42rem;
          margin: 0.75rem 0 0;
          color: var(--detector-muted);
          font-size: 1.06rem;
          line-height: 1.6;
        }
        .limitation {
          margin: 0.75rem 0 0;
          color: var(--detector-muted);
          font-size: 0.92rem;
          font-weight: 650;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          padding: 0.5rem;
        }
        .panel-heading {
          margin-bottom: 1rem;
        }
        .panel-heading h2,
        .report-empty h2,
        .section-heading h2 {
          margin: 0.2rem 0 0;
          color: var(--detector-ink);
          font-size: 1.35rem;
          letter-spacing: -0.025em;
        }
        .preview-empty {
          display: grid;
          min-height: 19rem;
          place-items: center;
          border: 1px dashed var(--detector-line);
          border-radius: var(--detector-radius);
          background: var(--detector-surface-muted);
          color: var(--detector-muted);
          text-align: center;
        }
        .preview-empty strong {
          display: block;
          color: var(--detector-ink);
          font-size: 1.08rem;
        }
        .preview-empty p,
        .report-empty p:not(.eyebrow),
        .section-heading p:not(.eyebrow) {
          margin: 0.5rem 0 0;
          color: var(--detector-muted);
          line-height: 1.55;
        }
        .report-empty, .result-summary {
          margin-top: 1rem;
          padding: 1.4rem 1.5rem;
        }
        .report-empty--ready {
          background: #fbfaf6;
        }
        .result-summary--ai {
          background: var(--detector-danger-soft);
        }
        .result-summary--real {
          background: var(--detector-success-soft);
        }
        .result-summary h2 {
          margin: 0.15rem 0 0;
          color: var(--detector-ink);
          font-family: var(--detector-display);
          font-size: clamp(2rem, 4vw, 3rem);
          letter-spacing: -0.04em;
        }
        .result-summary--ai h2 { color: var(--detector-danger); }
        .result-summary--real h2 { color: var(--detector-success); }
        .metric-group-label {
          margin: 1.25rem 0 0.5rem;
        }
        div[data-testid="stMetric"] {
          padding: 0.9rem 1rem;
        }
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] *,
        small {
          color: var(--detector-muted) !important;
        }
        .stButton > button {
          border: 1px solid var(--detector-button-bg);
          border-radius: 0.75rem;
          background: var(--detector-button-bg);
          color: var(--detector-button-text) !important;
          font-weight: 750;
        }
        .stButton > button *, .stButton > button p {
          color: var(--detector-button-text) !important;
          font-weight: 750;
        }
        .stButton > button:disabled,
        .stButton > button[disabled] {
          border-color: var(--detector-line);
          background: var(--detector-button-disabled-bg);
          color: var(--detector-button-disabled-text) !important;
          opacity: 1;
        }
        .stButton > button:disabled *,
        .stButton > button:disabled p,
        .stButton > button[disabled] p,
        .stButton > button[disabled] * {
          color: var(--detector-button-disabled-text) !important;
        }
        div[data-testid="stFileUploader"] button {
          border-color: var(--detector-line);
          background: var(--detector-surface-muted);
          color: var(--detector-ink) !important;
        }
        div[data-testid="stFileUploader"] button *,
        div[data-testid="stFileUploader"] button p {
          color: var(--detector-ink) !important;
        }
        div[data-testid="stFileUploader"] section,
        div[data-baseweb="select"] > div {
          border-color: var(--detector-line);
          background: var(--detector-surface);
        }
        div[data-testid="stImage"] img {
          border: 1px solid var(--detector-line);
          border-radius: 0.875rem;
        }
        div[data-testid="stAlert"] {
          border: 1px solid var(--detector-line);
          border-radius: 0.875rem;
        }
        @media (max-width: 920px) {
          .preview-empty {
            min-height: 14rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
