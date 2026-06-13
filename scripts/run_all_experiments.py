from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SMOKE_ARGS = [
    ["-m", "src.data.make_dummy_dataset", "--num_real", "30", "--num_fake", "30", "--output_dir", "data/raw/dummy", "--csv", "data/metadata/dataset.csv"],
    ["-m", "src.data.validate_metadata", "--csv", "data/metadata/dataset.csv"],
    ["-m", "src.features.cache_features", "--config", "configs/default.yaml", "--feature_type", "frequency", "--split", "train"],
    ["-m", "src.features.cache_features", "--config", "configs/default.yaml", "--feature_type", "frequency", "--split", "val"],
    ["-m", "src.features.cache_features", "--config", "configs/default.yaml", "--feature_type", "frequency", "--split", "test"],
    ["-m", "src.train.train_frequency", "--config", "configs/default.yaml"],
    ["-m", "src.eval.evaluate", "--config", "configs/default.yaml", "--model", "frequency_only", "--split", "test"],
    ["-m", "src.eval.robustness", "--config", "configs/default.yaml", "--model", "frequency_only", "--split", "test"],
]

DEPRECATION = """
DEPRECATED: scripts/run_all_experiments.py no longer runs the legacy experiment stack.
It delegates to the current CPU-safe frequency smoke sequence documented in README.md when run without flags.
For direct use, prefer these python -m src... commands:
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(DEPRECATION, file=sys.stderr)
    for command_args in SMOKE_ARGS:
        print("  python " + " ".join(command_args), file=sys.stderr)
    print("  streamlit run src/app/app.py", file=sys.stderr)
    if any(arg in {"-h", "--help"} for arg in args):
        return 0
    if args:
        print("Legacy flags are no longer supported by this wrapper; use README.md target commands instead.", file=sys.stderr)
        return 2
    for command_args in SMOKE_ARGS:
        command = [sys.executable, *command_args]
        print("running: " + " ".join(command))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    print("Current Streamlit app: streamlit run src/app/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
