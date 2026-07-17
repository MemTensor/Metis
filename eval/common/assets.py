"""Resolve public model IDs and release-local checkpoint assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.common.paths import REPO_ROOT


DEFAULT_ASSET_REGISTRY = REPO_ROOT / "eval/configs/assets.json"
DEFAULT_DATA_MANIFEST = REPO_ROOT / "eval/data/manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_assets(path: Path = DEFAULT_ASSET_REGISTRY) -> dict[str, str]:
    payload = load_json(path)
    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise ValueError(f"{path} must contain an object named 'assets'")
    return {str(key): str(value) for key, value in assets.items()}


def resolve_asset(asset_id: str, assets: dict[str, str]) -> str:
    if asset_id not in assets:
        raise KeyError(f"asset {asset_id!r} is missing from eval/configs/assets.json")
    value = assets[asset_id]
    if value.startswith(("eval/artifacts/", "artifacts/", "./", "../")):
        return str((REPO_ROOT / value).resolve())
    return value


def dataset_map(path: Path = DEFAULT_DATA_MANIFEST) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {item["id"]: item for item in payload["files"]}


def dataset_path(dataset_id: str, data_dir: Path, manifest: Path = DEFAULT_DATA_MANIFEST) -> Path:
    items = dataset_map(manifest)
    if dataset_id not in items:
        raise KeyError(f"dataset {dataset_id!r} is missing from {manifest}")
    return data_dir / items[dataset_id]["path"]
