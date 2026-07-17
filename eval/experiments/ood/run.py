#!/usr/bin/env python3
"""Run and score one paper Table 8 ATM-Bench or MemDaily cell."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from eval.common.cli import add_common_arguments, read_config, write_json
from eval.common.judge import DEFAULT_BASE_URL, DEFAULT_MODEL


DATASETS = ("atm", "memdaily")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--dataset", action="append", choices=DATASETS, default=[])
    parser.add_argument("--stage", choices=("inference", "score", "all"), default="all")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--model-label", default="paper_eval")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device-map", default="single", choices=("single", "paired_layers", "auto", "balanced"))
    parser.add_argument("--model-parallel-devices", default="")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--judge-max-attempts", type=int, default=12)
    parser.add_argument("--judge-retry-sleep", type=float, default=1.0)
    parser.add_argument("--judge-retry-backoff", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def cell_commands(args: argparse.Namespace, dataset: str) -> list[list[str]]:
    instances = args.data_dir / f"{dataset}.jsonl"
    cell = args.output_dir / dataset
    raw = cell / "raw.jsonl"
    commands: list[list[str]] = []
    if args.stage in {"inference", "all"}:
        if args.method in {"metis", "temp_lora", "delta_mem"}:
            model_path = args.checkpoint if args.method == "metis" else args.model
            command = [
                sys.executable,
                "-m",
                "eval.benchmarks.ood.scripts.run_memory_only",
                "--method",
                args.method,
                "--model-path",
                model_path or "<MODEL_OR_CHECKPOINT_REQUIRED>",
                "--model-label",
                args.model_label,
                "--instances",
                str(instances),
                "--output",
                str(raw),
                "--run-id",
                f"table8_{dataset}_{args.method}",
                "--device",
                args.device,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--query-style",
                "minimal",
                "--limit",
                str(args.limit),
            ]
            if args.method in {"metis", "temp_lora"}:
                command.extend(["--device-map", args.device_map])
                if args.method == "metis" and args.model_parallel_devices:
                    command.extend(["--model-parallel-devices", args.model_parallel_devices])
                for item in args.max_memory:
                    command.extend(["--max-memory", item])
            if args.method == "delta_mem":
                command.extend(["--adapter-dir", args.adapter_dir or "<ADAPTER_REQUIRED>"])
            if args.method == "temp_lora":
                command.extend(["--seed", str(args.seed)])
            commands.append(command)
        else:
            raise ValueError(f"Unsupported Table 8 method: {args.method}")
    if args.stage in {"score", "all"}:
        commands.append(
            [
                sys.executable,
                "-m",
                "eval.benchmarks.ood.scripts.score_official_gold",
                "--instances",
                str(instances),
                "--input",
                str(raw),
                "--output",
                str(cell / "scored.jsonl"),
                "--summary",
                str(cell / "summary.json"),
                "--judge-base-url",
                args.judge_base_url,
                "--judge-model",
                args.judge_model,
                "--api-key-env",
                args.api_key_env,
                "--limit",
                str(args.limit),
                "--concurrency",
                str(args.judge_concurrency),
                "--judge-repeats",
                "3",
                "--judge-max-attempts",
                str(args.judge_max_attempts),
                "--judge-retry-sleep",
                str(args.judge_retry_sleep),
                "--judge-retry-backoff",
                str(args.judge_retry_backoff),
            ]
        )
    return commands


def main() -> None:
    args = parse_args()
    if not args.data_dir:
        raise ValueError("--data-dir is required")
    datasets = tuple(args.dataset) if args.dataset else DATASETS
    commands = [command for dataset in datasets for command in cell_commands(args, dataset)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "commands.json",
        {
            "experiment": "table_8_ood",
            "config": read_config(args.config),
            "commands": commands,
        },
    )
    if args.dry_run:
        return
    for dataset in datasets:
        instances = args.data_dir / f"{dataset}.jsonl"
        if not instances.is_file():
            raise FileNotFoundError(instances)
    if args.method in {"temp_lora", "delta_mem"} and not args.model:
        raise ValueError(f"--model is required for {args.method}")
    if args.method == "metis" and not args.checkpoint:
        raise ValueError("--checkpoint is required for metis")
    if args.method == "delta_mem" and not args.adapter_dir:
        raise ValueError("--adapter-dir is required for delta_mem")
    for command in commands:
        env = None
        if args.method == "temp_lora":
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(args.seed)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
