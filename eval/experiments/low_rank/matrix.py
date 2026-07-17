#!/usr/bin/env python3
"""Run the paper's Metis-4B, 28-cell LowRankMemory workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval.common.assets import DEFAULT_ASSET_REGISTRY, load_assets, resolve_asset
from eval.common.cli import read_config, write_json
from eval.common.paths import EVAL_ROOT
from eval.data.verify import verify


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_10_low_rank.json"
DEFAULT_DATA_DIR = EVAL_ROOT / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_REGISTRY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--rank", action="append", default=[])
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--stage", choices=("raw", "score", "score-watch", "summary", "all"), default="all")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--raw-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge-concurrency", type=int, default=64)
    parser.add_argument("--judge-base-url", default="https://api.openai.com")
    parser.add_argument("--judge-model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = read_config(args.config)
    models = config["models"]
    if args.model:
        by_id = {item["id"]: item for item in models}
        unknown = sorted(set(args.model) - set(by_id))
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
        models = [by_id[item] for item in args.model]
    selected_ranks = config["ranks"]
    if args.rank:
        requested = {"full" if value == "full" else int(value) for value in args.rank}
        unknown = requested - set(config["ranks"])
        if unknown:
            raise ValueError(f"unknown ranks: {sorted(map(str, unknown))}")
        selected_ranks = [value for value in config["ranks"] if value in requested]
    benchmark_ids = [item["id"] for item in config["benchmarks"]]
    if args.benchmark:
        unknown = sorted(set(args.benchmark) - set(benchmark_ids))
        if unknown:
            raise ValueError(f"unknown benchmarks: {unknown}")
        benchmark_ids = args.benchmark
    runner_names = {item["id"]: item["runner_name"] for item in config["benchmarks"]}
    judge_model = args.judge_model or config["scoring"]["judge_model"]
    ranks = ",".join(str(value) for value in selected_ranks)
    assets = load_assets(args.assets)
    commands = []
    for model in models:
        commands.append(
            [
                sys.executable,
                "-m",
                "eval.experiments.low_rank.run",
                "--output-dir",
                str(args.output_dir / "models" / model["id"]),
                "--run-id",
                f"lowrank_{model['id']}",
                "--checkpoint",
                resolve_asset(model["checkpoint"], assets),
                "--data-dir",
                str(args.data_dir),
                "--model-label-prefix",
                model["model_label_prefix"],
                "--ranks",
                ranks,
                "--stage",
                args.stage,
                "--gpu-ids",
                args.gpu_ids,
                "--raw-workers",
                str(args.raw_workers),
                "--device",
                args.device,
                "--limit",
                str(args.limit),
                "--judge-concurrency",
                str(args.judge_concurrency),
                "--judge-base-url",
                args.judge_base_url,
                "--judge-model",
                judge_model,
                "--judge-api-key-env",
                args.api_key_env,
                "--judge-repeats",
                str(config["scoring"]["judge_repeats"]),
            ]
        )
        for benchmark_id in benchmark_ids:
            commands[-1] += ["--benchmark", runner_names[benchmark_id]]
    cell_count = len(models) * len(selected_ranks) * len(benchmark_ids)
    if not args.model and not args.rank and not args.benchmark and cell_count != config["expected_cells"]:
        raise RuntimeError(f"matrix expanded to {cell_count}, expected {config['expected_cells']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "experiment": config["experiment"],
            "protocol_status": config["protocol_status"],
            "result_status": config["result_status"],
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
    data_report = verify(args.data_dir, only=set(benchmark_ids))
    if not data_report["ok"]:
        raise RuntimeError("LowRankMemory data failed eval/data/manifest.json verification")
    for command in commands:
        subprocess.run(command, check=True)
    if args.stage in {"score", "summary", "all"} and not args.model and not args.rank and not args.benchmark and not args.limit:
        subprocess.run(
            [sys.executable, "-m", "eval.experiments.low_rank.audit", "--run-dir", str(args.output_dir), "--data-dir", str(args.data_dir), "--config", str(args.config), "--assets", str(args.assets)],
            check=True,
        )


if __name__ == "__main__":
    main()
