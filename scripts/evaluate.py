from __future__ import annotations

import sys

GUIDANCE = """
DEPRECATED: scripts/evaluate.py refreshes legacy experiment-directory metrics.

Use the current cache/checkpoint evaluator instead:
  python -m src.eval.evaluate --config configs/default.yaml --model frequency_only --split test

Optional models are clip_only and fusion only after their .npy caches and PyTorch checkpoints exist. See README.md for the target evaluation workflow.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(GUIDANCE, file=sys.stderr)
    return 0 if any(arg in {"-h", "--help"} for arg in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
