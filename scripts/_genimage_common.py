from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def manifest_rows(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if split:
        rows=[row for row in rows if row.get('split') == split]
    return rows


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row['label']) for row in rows], dtype=np.int64)


def path_for(row: dict[str, str]) -> str:
    return row.get('path') or row.get('filepath') or str(Path(row.get('root','')) / row.get('rel_path',''))
