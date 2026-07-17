"""Repository-relative paths used by all evaluation entry points."""

from __future__ import annotations

import os
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent


def data_root(override: str | Path | None = None) -> Path:
    """Resolve the external benchmark-data root without an internal default."""

    value = override or os.environ.get("METIS_EVAL_DATA")
    if not value:
        raise ValueError("Pass --data-dir or set METIS_EVAL_DATA.")
    return Path(value).expanduser().resolve()


def output_root(override: str | Path | None = None) -> Path:
    """Resolve a generated-output root, which must remain outside tracked results."""

    value = override or os.environ.get("METIS_EVAL_OUTPUT")
    if not value:
        raise ValueError("Pass --output-dir or set METIS_EVAL_OUTPUT.")
    return Path(value).expanduser().resolve()
