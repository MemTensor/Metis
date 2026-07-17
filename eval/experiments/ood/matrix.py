#!/usr/bin/env python3
"""Expand and run the paper's 14-cell ATM/MemDaily OOD protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.common.assets import DEFAULT_ASSET_REGISTRY, load_assets, resolve_asset
from eval.common.cli import read_config, write_json
from eval.common.paths import EVAL_ROOT
from eval.data.verify import verify


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_8_ood.json"
DEFAULT_DATA_DIR = EVAL_ROOT / "data"


def select(items: list[Any], requested: list[str]) -> list[Any]:
    if not requested:
        return items
    if items and isinstance(items[0], dict):
        by_id = {item["id"]: item for item in items}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError(f"unknown IDs: {missing}")
        return [by_id[item] for item in requested]
    missing = sorted(set(requested) - set(items))
    if missing:
        raise ValueError(f"unknown IDs: {missing}")
    return [item for item in requested]


def method_command(method: dict[str, Any], datasets: list[str], args: argparse.Namespace, assets: dict[str, str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eval.experiments.ood.run",
        "--config",
        str(args.config),
        "--data-dir",
        str(args.data_dir / "ood"),
        "--output-dir",
        str(args.output_dir / "cells" / method["id"]),
        "--method",
        method["implementation"],
        "--model-label",
        method["id"],
        "--stage",
        args.stage,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--judge-base-url",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
        "--api-key-env",
        args.api_key_env,
    ]
    for dataset in datasets:
        command += ["--dataset", dataset]
    if "model" in method:
        command += ["--model", resolve_asset(method["model"], assets)]
    if "checkpoint" in method:
        command += ["--checkpoint", resolve_asset(method["checkpoint"], assets)]
    if "adapter" in method:
        command += ["--adapter-dir", resolve_asset(method["adapter"], assets)]
    if "device_map" in method:
        command += ["--device-map", method["device_map"]]
    if "model_parallel_devices" in method:
        command += ["--model-parallel-devices", method["model_parallel_devices"]]
    for item in method.get("max_memory", []):
        command += ["--max-memory", item]
    if "seed" in method:
        command += ["--seed", str(method["seed"])]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_REGISTRY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("inference", "score", "all"), default="all")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--judge-base-url", default="https://api.openai.com")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = read_config(args.config)
    methods = select(config["methods"], args.method)
    datasets = select(config["datasets"], args.dataset)
    cell_count = len(methods) * len(datasets)
    if not args.method and not args.dataset and cell_count != config["expected_cells"]:
        raise RuntimeError(f"matrix expanded to {cell_count}, expected {config['expected_cells']}")
    assets = load_assets(args.assets)
    commands = [method_command(method, datasets, args, assets) for method in methods]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "experiment": config["experiment"],
            "status": config["status"],
            "cell_count": cell_count,
            "config": str(args.config),
            "assets": str(args.assets),
            "data_dir": str(args.data_dir),
            "commands": commands,
        },
    )
    if args.dry_run:
        print(json.dumps({"cell_count": cell_count, "run_config": str(args.output_dir / 'run_config.json')}))
        return
    data_report = verify(args.data_dir, only=set(datasets))
    if not data_report["ok"]:
        raise RuntimeError("selected OOD data failed eval/data/manifest.json verification")
    for command in commands:
        subprocess.run(command, check=True)
    if args.stage in {"score", "all"} and not args.method and not args.dataset and not args.limit:
        subprocess.run(
            [sys.executable, "-m", "eval.experiments.ood.audit", "--run-dir", str(args.output_dir), "--data-dir", str(args.data_dir), "--config", str(args.config), "--assets", str(args.assets)],
            check=True,
        )


if __name__ == "__main__":
    main()
