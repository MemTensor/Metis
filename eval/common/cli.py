"""Uniform CLI options and JSON/JSONL utilities for paper runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="Backbone path or public model identifier.")
    parser.add_argument("--checkpoint", help="Metis checkpoint directory.")
    parser.add_argument("--data-dir", type=Path, help="External benchmark-data root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", default="metis")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
