from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def dump_snapshot(config: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'config_snapshot.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
