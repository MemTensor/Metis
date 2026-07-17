#!/usr/bin/env python3
"""Expand and run the canonical 77-cell main-table matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval.common.assets import DEFAULT_ASSET_REGISTRY, dataset_path, load_assets, resolve_asset
from eval.common.cli import read_config, write_json
from eval.common.paths import EVAL_ROOT
from eval.data.verify import verify
from eval.experiments.main_tables.run import (
    expected_max_input_tokens,
    expected_max_new_tokens,
)


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_5_6_main.json"
DEFAULT_DATA_DIR = EVAL_ROOT / "data"


def selected(values: list[dict[str, Any]], requested: list[str]) -> list[dict[str, Any]]:
    if not requested:
        return values
    by_id = {item["id"]: item for item in values}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"unknown IDs: {missing}")
    return [by_id[item] for item in requested]


def expand_cells(config: dict[str, Any], benchmarks: list[str], methods: list[str]) -> list[dict[str, Any]]:
    cells = []
    for method in selected(config["methods"], methods):
        skipped = set(method.get("skip_benchmarks", []))
        for benchmark in selected(config["benchmarks"], benchmarks):
            if benchmark["id"] not in skipped:
                cells.append({"method": method, "benchmark": benchmark})
    return cells


def cell_command(
    cell: dict[str, Any],
    args: argparse.Namespace,
    assets: dict[str, str],
    config: dict[str, Any],
) -> list[str]:
    method = cell["method"]
    benchmark = cell["benchmark"]
    command = [
        sys.executable,
        "-m",
        "eval.experiments.main_tables.run",
        "--config",
        str(args.config),
        "--benchmark",
        benchmark["runner_id"],
        "--instances",
        str(dataset_path(benchmark["id"], args.data_dir)),
        "--method",
        method["implementation"],
        "--model-label",
        method["id"],
        "--output-dir",
        str(args.output_dir / "cells" / f"{method['id']}__{benchmark['id']}"),
        "--stage",
        args.stage,
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        "--max-new-tokens",
        str(
            expected_max_new_tokens(
                config, method["implementation"], benchmark["runner_id"]
            )
        ),
        "--max-input-tokens",
        str(
            expected_max_input_tokens(
                config, method["implementation"], benchmark["runner_id"]
            )
        ),
        "--judge-concurrency",
        str(args.judge_concurrency),
        "--judge-base-url",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
        "--api-key-env",
        args.api_key_env,
    ]
    if "model" in method:
        command += ["--model", resolve_asset(method["model"], assets)]
    if "checkpoint" in method:
        command += ["--checkpoint", resolve_asset(method["checkpoint"], assets)]
    if "adapter" in method:
        command += ["--adapter-dir", resolve_asset(method["adapter"], assets)]
    if "embedding_model" in method:
        command += ["--embedding-model", resolve_asset(method["embedding_model"], assets)]
    if "device_map" in method:
        command += ["--device-map", method["device_map"]]
    if "model_parallel_devices" in method:
        command += ["--model-parallel-devices", method["model_parallel_devices"]]
    if "max_memory" in method:
        command += ["--max-memory", *method["max_memory"]]
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
    parser.add_argument("--benchmark", action="append", default=[])
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
    assets = load_assets(args.assets)
    cells = expand_cells(config, args.benchmark, args.method)
    if not args.benchmark and not args.method and len(cells) != config["expected_cells"]:
        raise RuntimeError(f"matrix expanded to {len(cells)}, expected {config['expected_cells']}")
    commands = [cell_command(cell, args, assets, config) for cell in cells]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "experiment": config["experiment"],
            "status": config["status"],
            "config": str(args.config),
            "assets": str(args.assets),
            "data_dir": str(args.data_dir),
            "cell_count": len(cells),
            "commands": commands,
        },
    )
    if args.dry_run:
        print(json.dumps({"cell_count": len(cells), "run_config": str(args.output_dir / 'run_config.json')}))
        return
    data_report = verify(args.data_dir, only={cell["benchmark"]["id"] for cell in cells})
    if not data_report["ok"]:
        raise RuntimeError("selected main-table data failed eval/data/manifest.json verification")
    for command in commands:
        subprocess.run(command, check=True)
    if args.stage == "all" and not args.benchmark and not args.method and not args.limit:
        subprocess.run(
            [sys.executable, "-m", "eval.experiments.main_tables.audit", "--run-dir", str(args.output_dir), "--data-dir", str(args.data_dir), "--config", str(args.config), "--assets", str(args.assets)],
            check=True,
        )


if __name__ == "__main__":
    main()
