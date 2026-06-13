from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/run_robustness.py runs robustness on legacy manifest-v1 experiment artifacts.

Use the current checkpoint-backed robustness evaluator instead:
  python -m src.eval.robustness --config configs/default.yaml --model frequency_only --split test

This mandatory path is CPU-safe and frequency-only. Optional clip_only/fusion robustness requires prepared CLIP runtime artifacts.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
