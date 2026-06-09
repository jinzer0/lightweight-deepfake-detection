from __future__ import annotations

# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    LIMITATION_WARNING,
    ImagePredictor,
    PredictorArtifactError,
    UnsupportedPredictorArtifactError,
)


REQUIRED_ARTIFACTS = ("config.yaml", "model.joblib", "scaler.joblib")
UPLOAD_TYPES = ("jpg", "jpeg", "png")
DEFAULT_ARTIFACT_SEARCH_ROOTS = ("outputs", "artifacts", "experiments", "runs")
WARNING_TEXT = " ".join(LIMITATION_WARNING.splitlines())


class UploadedImage(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


def main() -> None:
    st.set_page_config(page_title="AI Image Detector Demo", page_icon="D", layout="wide")
    _apply_design_tokens()

    st.markdown(
        """
        <section class="hero-card">
          <p class="eyebrow">Experimental artifact-backed demo</p>
          <h1>AI Image Detector</h1>
          <p class="lede">Upload one JPG or PNG and run it against a saved probability-capable artifact.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.warning(WARNING_TEXT)

    artifact_options = _artifact_options(PROJECT_ROOT)
    with st.sidebar:
        st.header("Prediction setup")
        selected_artifact = st.selectbox(
            "Discovered artifact directory",
            options=artifact_options,
            index=0,
            help="Directories are listed only when config.yaml, model.joblib, and scaler.joblib are present.",
        )
        default_path = "" if selected_artifact == "Manual path" else selected_artifact
        artifact_path = st.text_input(
            "Artifact directory",
            value=default_path,
            placeholder="outputs/frequency_lr",
            help="Use a frequency_only LogisticRegression artifact with probability prediction enabled.",
        )
        threshold = st.slider("Fake probability threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    upload_column, result_column = st.columns([0.92, 1.08], gap="large")
    with upload_column:
        st.subheader("Image")
        uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=UPLOAD_TYPES)
        st.caption("The app passes the uploaded file to the shared predictor path; preprocessing stays inside `src.inference.predictor`.")

    with result_column:
        st.subheader("Result")
        if not uploaded_file:
            st.info("Upload an image to enable an artifact-backed prediction.")
            return
        if not artifact_path.strip():
            st.error("Enter an artifact directory containing config.yaml, model.joblib, and scaler.joblib.")
            return

        if st.button("Run prediction", type="primary", use_container_width=True):
            _run_prediction(uploaded_file, artifact_path.strip(), threshold)


def _run_prediction(uploaded_file: UploadedImage, artifact_path: str, threshold: float) -> None:
    experiment_dir = _resolve_artifact_path(artifact_path)
    try:
        predictor = ImagePredictor.from_experiment_dir(experiment_dir)
        suffix = _upload_suffix(uploaded_file)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(uploaded_file.getvalue())
        try:
            prediction = predictor.predict(temp_path, threshold=threshold)
        finally:
            temp_path.unlink(missing_ok=True)
    except UnsupportedPredictorArtifactError as exc:
        st.error(_actionable_artifact_error(str(exc)))
        return
    except PredictorArtifactError as exc:
        st.error(_actionable_artifact_error(str(exc)))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    decision = "Likely AI-generated" if prediction.pred_label == 1 else "Likely real"
    st.metric("prob_fake", f"{prediction.prob_fake:.4f}")
    st.metric("Threshold decision", decision)
    st.write(
        {
            "score": round(float(prediction.score), 6),
            "label": prediction.label_text,
            "threshold": round(float(prediction.effective_threshold), 4),
        }
    )
    for warning in prediction.warnings:
        st.warning(" ".join(str(warning).splitlines()))


def _resolve_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _upload_suffix(uploaded_file: object) -> str:
    name = cast(str, getattr(uploaded_file, "name", "upload.png"))
    suffix = Path(str(name)).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png"} else ".png"


def _artifact_options(root: Path) -> list[str]:
    discovered = sorted(str(path.relative_to(root)) for path in _discover_artifact_dirs(root))
    return ["Manual path", *discovered]


def _discover_artifact_dirs(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    search_roots = [root / name for name in DEFAULT_ARTIFACT_SEARCH_ROOTS if (root / name).is_dir()]
    if all((root / file_name).is_file() for file_name in REQUIRED_ARTIFACTS):
        search_roots.append(root)
    for search_root in search_roots:
        for config_path in search_root.rglob("config.yaml"):
            candidate = config_path.parent
            if candidate not in seen and all((candidate / file_name).is_file() for file_name in REQUIRED_ARTIFACTS):
                seen.add(candidate)
                yield candidate


def _actionable_artifact_error(message: str) -> str:
    lowered = message.lower()
    if "does not exist" in lowered:
        return f"Artifact directory was not found. Check the path and try again. Details: {message}"
    if "missing required artifact" in lowered:
        return f"Artifact directory is incomplete. It must contain config.yaml, model.joblib, and scaler.joblib. Details: {message}"
    if "linear svm" in lowered or "decision_score_only" in lowered or "score-only" in lowered:
        return f"This artifact is score-only and cannot produce prob_fake. Use a LogisticRegression probability artifact. Details: {message}"
    if "clip" in lowered or "fusion" in lowered:
        return f"CLIP and fusion artifacts are not supported for live probability inference yet. Use a frequency_only LogisticRegression artifact. Details: {message}"
    return f"Artifact is not compatible with this demo. Details: {message}"


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
          --detector-space-sm: 0.75rem;
          --detector-space-md: 1rem;
          --detector-space-lg: 1.5rem;
          --detector-space-xl: 2rem;
          --detector-radius-lg: 1.5rem;
        }
        .stApp {
          color: var(--detector-ink);
          background:
            radial-gradient(circle at 12% 12%, rgb(var(--detector-accent-rgb) / 0.18), transparent 30rem),
            linear-gradient(135deg, var(--detector-paper), var(--detector-haze) 48%, var(--detector-cream));
        }
        .block-container {
          padding-top: var(--detector-space-xl);
        }
        .hero-card {
          border: 1px solid var(--detector-line);
          border-radius: var(--detector-radius-lg);
          padding: var(--detector-space-xl);
          margin-bottom: var(--detector-space-lg);
          background: linear-gradient(140deg, rgb(var(--detector-panel-rgb) / 0.92), rgb(var(--detector-warm-rgb) / 0.72));
          box-shadow: var(--detector-shadow);
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
