from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_APP = PROJECT_ROOT / "src" / "app" / "app.py"

GUIDANCE = """
DEPRECATED: app/streamlit_app.py is a compatibility entrypoint.
Use the current Streamlit command instead:
  streamlit run src/app/app.py
""".strip()


def main() -> None:
    if _running_under_streamlit():
        print(GUIDANCE, file=sys.stderr)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        _ = runpy.run_path(TARGET_APP.as_posix(), run_name="__main__")
        return
    print(GUIDANCE, file=sys.stderr)


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


if __name__ == "__main__":
    main()
