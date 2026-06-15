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
LIMITATION_TEXT = "This result is a reference estimate from a limited dataset, not a guarantee that every AI-generated image can be detected."
MODEL_OPTIONS = ("frequency_only", "clip_only", "fusion")
MODEL_LABELS = {
    "frequency_only": "Fast texture check",
    "clip_only": "Visual similarity check",
    "fusion": "Balanced combined check",
}
MODEL_DESCRIPTIONS = {
    "frequency_only": "A quick scan focused on image texture patterns.",
    "clip_only": "A visual-language scan for broader image cues.",
    "fusion": "A combined scan that weighs multiple detector signals.",
}


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
          <h1>AI-GEN Image Detection model</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if LIMITATION_TEXT:
        st.info(LIMITATION_TEXT)

    with st.container(border=True):
        st.markdown(
            """
            <section class="flow-heading">
              <p class="eyebrow">Start here</p>
              <h2>Upload and analyze in one step</h2>
              <p>Keep the default option for the quickest review, or choose a broader check before analyzing.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

        action_column, preview_column = st.columns([1, 1], gap="large")
        with action_column:
            model_name = st.selectbox(
                "Review style",
                options=MODEL_OPTIONS,
                index=0,
                format_func=_model_label,
                help="Choose the kind of review to run. Some options may take longer depending on this app's setup.",
            )
            st.caption(MODEL_DESCRIPTIONS[str(model_name)])

            uploaded_file = st.file_uploader("Choose an image", type=UPLOAD_TYPES, accept_multiple_files=False)
            st.caption("Supported formats: JPG, JPEG, PNG.")

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
                    <section class="empty-state">
                      <strong>No image selected yet</strong>
                      <p>Upload one image to preview it here, then run the analysis from the button beside it.</p>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.image(uploaded_file, caption="Selected image", width="stretch")

    st.markdown(
        """
        <section class="section-heading section-heading--results">
          <p class="eyebrow">Results</p>
          <h2>Your image report</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if uploaded_file is None:
        st.markdown(
            """
            <section class="empty-state empty-state--wide">
              <strong>Waiting for an image</strong>
              <p>Your result will appear here after you choose an image and start the analysis.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return
    if analyze_clicked:
        _run_prediction(uploaded_file, str(model_name))
    else:
        st.markdown(
            """
            <section class="empty-state empty-state--wide">
              <strong>Ready when you are</strong>
              <p>Use the analyze button above to generate the image report.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )


def _model_label(model_name: str) -> str:
    return MODEL_LABELS[model_name]


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
    branch_columns[0].metric("Visual similarity signal", _format_optional_score(result["clip_score"]))
    branch_columns[1].metric("Texture signal", _format_optional_score(result["frequency_score"]))
    branch_columns[2].metric("Combined signal", _format_optional_score(result["fusion_score"]))


def _display_visualizations(result: dict[str, float | str | None]) -> None:
    spectrum_path = result["spectrum_path"]
    radial_path = result["radial_spectrum_path"]
    st.divider()
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
        st.subheader("DCT/FFT spectrum")
        _display_image_path(spectrum_path)
    with viz_columns[1]:
        st.subheader("Radial spectrum")
        _display_image_path(radial_path)


def _display_image_path(path_value: float | str | None) -> None:
    if not isinstance(path_value, str) or not path_value:
        st.info("This view is not available for the selected review.")
        return
    image_path = Path(path_value)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    if not image_path.is_file():
        st.info("This view is not available for the selected review.")
        return
    st.image(image_path.as_posix(), width="stretch")


def _format_optional_score(value: float | str | None) -> str:
    if value is None:
        return "Not available"
    return _format_score(_as_float(value))


def _format_score(value: float) -> str:
    return f"{value:.4f}"


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
          --detector-ink: #20201d;
          --detector-muted: #514d46;
          --detector-soft: #5f5a52;
          --detector-paper: #f7f5f0;
          --detector-paper-alt: #efede6;
          --detector-panel: #fffdf8;
          --detector-panel-muted: #f3f0e8;
          --detector-line: #ded8cc;
          --detector-line-strong: #c8c0b2;
          --detector-accent: #344a40;
          --detector-accent-hover: #24352d;
          --detector-accent-soft: #dce6df;
          --detector-accent-ink: #1f332a;
          --detector-focus: #0f5132;
          --detector-button-text: #fffaf0;
          --detector-button-disabled-bg: #d8d1c4;
          --detector-button-disabled-text: #514d46;
          --detector-button-disabled-border: #b8ae9f;
          --detector-warning: #5f4711;
          --detector-warning-soft: #f4ead4;
          --detector-danger-soft: #f1ded8;
          --detector-danger-ink: #7a3c2f;
          --detector-success-soft: #dfe9df;
          --detector-success-ink: #35543b;
          --detector-ink-rgb: 32 32 29;
          --detector-panel-rgb: 255 253 248;
          --detector-accent-rgb: 52 74 64;
          --detector-border-width: 1px;
          --detector-shadow-sm: 0 0.5rem 1.5rem rgb(var(--detector-ink-rgb) / 0.05);
          --detector-shadow-md: 0 1rem 2.75rem rgb(var(--detector-ink-rgb) / 0.08);
          --detector-space-2xs: 0.25rem;
          --detector-space-xs: 0.5rem;
          --detector-space-sm: 0.75rem;
          --detector-space-md: 1rem;
          --detector-space-lg: 1.5rem;
          --detector-space-xl: 2rem;
          --detector-space-2xl: 3rem;
          --detector-radius-sm: 0.625rem;
          --detector-radius-md: 0.875rem;
          --detector-radius-lg: 1.25rem;
          --detector-radius-pill: 999px;
          --detector-font-body: 'Aptos', 'IBM Plex Sans', 'Segoe UI', sans-serif;
          --detector-font-display: 'Iowan Old Style', 'Charter', Georgia, serif;
          --detector-text-xs: 0.78rem;
          --detector-text-sm: 0.92rem;
          --detector-text-md: 1rem;
          --detector-text-lg: 1.08rem;
          --detector-text-xl: 1.35rem;
          --detector-text-display: clamp(2.4rem, 5vw, 4.6rem);
          --detector-text-result: clamp(2rem, 4vw, 3.4rem);
          --detector-leading-tight: 0.98;
          --detector-leading-body: 1.65;
          --detector-tracking-tight: -0.03em;
          --detector-tracking-tighter: -0.045em;
          --detector-tracking-slight: 0.02em;
          --detector-tracking-wide: 0.1em;
          --detector-tracking-wider: 0.12em;
          --detector-weight-bold: 700;
          --detector-lift: -0.0625rem;
          --detector-transition: 160ms ease;
          --detector-max-width: 76rem;
          --detector-max-width-hero: 54rem;
          --detector-max-width-copy: 44rem;
        }
        .stApp {
          color: var(--detector-ink);
          background: linear-gradient(180deg, var(--detector-paper), var(--detector-paper-alt));
          font-family: var(--detector-font-body);
        }
        .block-container {
          max-width: var(--detector-max-width);
          padding-top: var(--detector-space-2xl);
          padding-bottom: var(--detector-space-2xl);
        }
        h1, h2, h3, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          color: var(--detector-ink);
          font-family: var(--detector-font-display);
          letter-spacing: var(--detector-tracking-tight);
        }
        p, label, span, div[data-testid="stMarkdownContainer"] {
          font-family: var(--detector-font-body);
        }
        .hero-card {
          border: var(--detector-border-width) solid var(--detector-line);
          border-radius: var(--detector-radius-lg);
          padding: var(--detector-space-2xl);
          margin-bottom: var(--detector-space-lg);
          background: rgb(var(--detector-panel-rgb) / 0.92);
          box-shadow: var(--detector-shadow-md);
        }
        .eyebrow {
          margin: 0 0 var(--detector-space-sm);
          color: var(--detector-accent-ink);
          font-size: var(--detector-text-xs);
          font-weight: var(--detector-weight-bold);
          letter-spacing: var(--detector-tracking-wider);
          text-transform: uppercase;
        }
        .hero-card h1 {
          max-width: var(--detector-max-width-hero);
          margin: 0;
          color: var(--detector-ink);
          font-family: var(--detector-font-display);
          font-size: var(--detector-text-display);
          line-height: var(--detector-leading-tight);
          letter-spacing: var(--detector-tracking-tighter);
        }
        .lede {
          max-width: var(--detector-max-width-copy);
          margin: var(--detector-space-md) 0 0;
          color: var(--detector-muted);
          font-size: var(--detector-text-lg);
          line-height: var(--detector-leading-body);
        }
        .section-heading, .flow-heading {
          margin-bottom: var(--detector-space-md);
        }
        .section-heading--compact {
          margin-top: var(--detector-space-sm);
        }
        .section-heading--results {
          margin: var(--detector-space-xl) 0 var(--detector-space-md);
        }
        .section-heading h2, .flow-heading h2 {
          margin: 0;
          font-size: var(--detector-text-xl);
          line-height: var(--detector-leading-tight);
        }
        .section-heading p:not(.eyebrow), .flow-heading p:not(.eyebrow) {
          margin: var(--detector-space-xs) 0 0;
          color: var(--detector-muted);
          font-size: var(--detector-text-sm);
          line-height: var(--detector-leading-body);
        }
        .empty-state, .result-summary {
          border: var(--detector-border-width) solid var(--detector-line);
          border-radius: var(--detector-radius-lg);
          padding: var(--detector-space-xl);
          background: rgb(var(--detector-panel-rgb) / 0.82);
          box-shadow: var(--detector-shadow-sm);
        }
        .empty-state strong {
          color: var(--detector-ink);
          font-family: var(--detector-font-display);
          font-size: var(--detector-text-xl);
          letter-spacing: var(--detector-tracking-tight);
        }
        .empty-state p {
          margin: var(--detector-space-xs) 0 0;
          color: var(--detector-muted);
          line-height: var(--detector-leading-body);
        }
        .empty-state--wide {
          margin-top: 0;
        }
        .result-summary {
          margin-bottom: var(--detector-space-lg);
        }
        .result-summary--ai {
          background: linear-gradient(180deg, var(--detector-danger-soft), rgb(var(--detector-panel-rgb) / 0.86));
        }
        .result-summary--real {
          background: linear-gradient(180deg, var(--detector-success-soft), rgb(var(--detector-panel-rgb) / 0.86));
        }
        .result-summary h2 {
          margin: 0;
          font-family: var(--detector-font-display);
          font-size: var(--detector-text-result);
          line-height: var(--detector-leading-tight);
          letter-spacing: var(--detector-tracking-tighter);
        }
        .result-summary--ai h2 {
          color: var(--detector-danger-ink);
        }
        .result-summary--real h2 {
          color: var(--detector-success-ink);
        }
        .result-summary p:not(.eyebrow) {
          margin: var(--detector-space-sm) 0 0;
          color: var(--detector-muted);
          font-size: var(--detector-text-md);
        }
        .metric-group-label {
          margin: var(--detector-space-lg) 0 var(--detector-space-xs);
          color: var(--detector-soft);
          font-size: var(--detector-text-xs);
          font-weight: var(--detector-weight-bold);
          letter-spacing: var(--detector-tracking-wide);
          text-transform: uppercase;
        }
        div[data-testid="stMetric"] {
          border: var(--detector-border-width) solid var(--detector-line);
          border-radius: var(--detector-radius-md);
          padding: var(--detector-space-md);
          background: rgb(var(--detector-panel-rgb) / 0.78);
          box-shadow: var(--detector-shadow-sm);
        }
        div[data-testid="stMetricLabel"] p {
          color: var(--detector-muted);
          font-size: var(--detector-text-xs);
          letter-spacing: var(--detector-tracking-slight);
        }
        div[data-testid="stMetricValue"] {
          color: var(--detector-ink);
          font-family: var(--detector-font-display);
          letter-spacing: var(--detector-tracking-tight);
        }
        div[data-testid="stFileUploader"] section {
          border-color: var(--detector-line-strong);
          border-radius: var(--detector-radius-md);
          background: rgb(var(--detector-panel-rgb) / 0.72);
        }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] * {
          color: var(--detector-muted);
        }
        div[data-testid="stAlert"] {
          border: var(--detector-border-width) solid var(--detector-line);
          border-radius: var(--detector-radius-md);
          background: var(--detector-warning-soft);
          color: var(--detector-warning);
        }
        [data-testid="stSidebar"] {
          background: var(--detector-panel);
          border-right: var(--detector-border-width) solid var(--detector-line);
        }
        .sidebar-eyebrow {
          margin: var(--detector-space-xs) 0 var(--detector-space-sm);
          color: var(--detector-accent-ink);
          font-size: var(--detector-text-xs);
          font-weight: var(--detector-weight-bold);
          letter-spacing: var(--detector-tracking-wider);
          text-transform: uppercase;
        }
        .stButton > button {
          border: var(--detector-border-width) solid var(--detector-accent-ink);
          border-radius: var(--detector-radius-md);
          background: var(--detector-accent);
          color: var(--detector-button-text);
          font-weight: var(--detector-weight-bold);
          transition: transform var(--detector-transition), box-shadow var(--detector-transition), background var(--detector-transition), border-color var(--detector-transition);
          box-shadow: var(--detector-shadow-sm);
        }
        .stButton > button *, .stButton > button p {
          color: var(--detector-button-text);
        }
        .stButton > button:hover {
          transform: translateY(var(--detector-lift));
          border-color: var(--detector-accent-hover);
          background: var(--detector-accent-hover);
          box-shadow: var(--detector-shadow-md);
        }
        .stButton > button:focus-visible {
          outline: var(--detector-border-width) solid var(--detector-focus);
          outline-offset: var(--detector-space-2xs);
        }
        .stButton > button:disabled,
        .stButton > button:disabled:hover,
        .stButton > button[disabled] {
          transform: none;
          border-color: var(--detector-button-disabled-border);
          background: var(--detector-button-disabled-bg);
          color: var(--detector-button-disabled-text);
          box-shadow: none;
          cursor: not-allowed;
          opacity: 1;
        }
        .stButton > button:disabled *,
        .stButton > button:disabled p,
        .stButton > button[disabled] *,
        .stButton > button[disabled] p {
          color: var(--detector-button-disabled-text);
        }
        hr {
          border-color: var(--detector-line);
        }
        img {
          border-radius: var(--detector-radius-md);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
