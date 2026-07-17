#!/usr/bin/env python3
"""Dispatch one Table 5/6 benchmark-method cell to its canonical runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.common.cli import add_common_arguments, read_config, write_json
from eval.common.judge import DEFAULT_BASE_URL, DEFAULT_MODEL


MEMQA = {"locomo_gold", "nextmem_contextual_generation"}
MEMOPS = {"memops_full", "memops_gold", "metis_test"}
CONTEXT_METHODS = {"no_context", "full_context", "partial_context"}


def expected_max_new_tokens(config: dict, method: str, benchmark: str) -> int:
    generation = config["generation"]
    if method in CONTEXT_METHODS:
        suite = "memqa" if benchmark in MEMQA else "memops"
        return int(generation["context_methods_max_new_tokens"][suite])
    return int(generation["memory_methods_max_new_tokens"])


def expected_max_input_tokens(config: dict, method: str, benchmark: str) -> int:
    if method in {"no_context", "full_context"} and benchmark == "metis_test":
        return int(config["generation"]["metis_test_context_max_input_tokens"])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--benchmark", required=True, choices=sorted(MEMQA | MEMOPS))
    parser.add_argument("--instances", type=Path)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--model-label", default="paper_eval")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="single", choices=("single", "paired_layers", "auto", "balanced"))
    parser.add_argument("--model-parallel-devices", default="")
    parser.add_argument("--max-memory", nargs="*", default=[])
    parser.add_argument("--query-style", default="memory_direct", choices=("default", "memory_direct", "minimal"))
    parser.add_argument("--delta-attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--stage", choices=("inference", "score", "all"), default="inference")
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=1)
    parser.add_argument("--judge-max-attempts", type=int, default=18)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)
    parser.add_argument("--judge-retry-backoff", type=float, default=1.5)
    parser.add_argument("--judge-progress-every", type=int, default=25)
    return parser.parse_args()


def raw_output_path(args: argparse.Namespace) -> Path:
    if args.method in {"no_context", "full_context"}:
        safe_label = args.model_label.replace("/", "_").replace(":", "_").replace(" ", "_")
        return args.output_dir / f"{safe_label}.{args.method}.raw.jsonl"
    return args.output_dir / f"{args.benchmark}.{args.method}.raw.jsonl"


def module_and_args(args: argparse.Namespace, instances: Path) -> tuple[str, list[str]]:
    common = [
        "--instances", str(instances), "--device", args.device,
        "--dtype", args.dtype, "--limit", str(args.limit),
    ]
    model_loading = ["--device-map", args.device_map]
    if args.max_memory:
        model_loading += ["--max-memory", *args.max_memory]
    metis_loading = ["--device-map", args.device_map]
    if args.model_parallel_devices:
        metis_loading += ["--model-parallel-devices", args.model_parallel_devices]
    for item in args.max_memory:
        metis_loading += ["--max-memory", item]
    temp_lora_loading = ["--device-map", args.device_map]
    for item in args.max_memory:
        temp_lora_loading += ["--max-memory", item]
    task = "memqa" if args.benchmark in MEMQA else "memops"
    model = args.model or "<MODEL_REQUIRED>"
    checkpoint = args.checkpoint or "<CHECKPOINT_REQUIRED>"
    adapter_dir = args.adapter_dir or "<ADAPTER_REQUIRED>"
    if args.method in {"no_context", "full_context"}:
        module = f"eval.benchmarks.{task}.scripts." + ("run_base_context" if task == "memqa" else "run_qwen_plain_context")
        context_args = [
            "--model-path", model, "--model-label", args.model_label,
            "--output-dir", str(args.output_dir), "--conditions", args.method,
            "--max-new-tokens", str(args.max_new_tokens), *model_loading, *common,
        ]
        if task == "memops":
            context_args += ["--max-input-tokens", str(args.max_input_tokens)]
        return module, context_args
    output = args.output_dir / f"{args.benchmark}.{args.method}.raw.jsonl"
    if args.method == "partial_context":
        module = "eval.methods.dense_rag.scripts.run_memqa_dense_rag" if task == "memqa" else "eval.benchmarks.memops.scripts.run_memop_dense_rag"
        return module, [
            "--model-path", model, "--model-label", args.model_label,
            "--embedding-model", args.embedding_model, "--output", str(output),
            "--max-new-tokens", str(args.max_new_tokens), *model_loading, *common,
        ]
    if task == "memqa":
        if args.method == "metis":
            return "eval.benchmarks.memqa.scripts.run_metis_memqa", [
                "--checkpoint", checkpoint, "--model-label", args.model_label,
                "--output", str(output), "--query-style", args.query_style,
                "--max-new-tokens", str(args.max_new_tokens), *metis_loading, *common,
            ]
        module = f"eval.methods.{args.method}.scripts.run_memqa_{args.method}"
        method_args = ["--model-path", model, "--model-label", args.model_label, "--output", str(output)]
        if args.method == "delta_mem":
            method_args += [
                "--adapter-dir", adapter_dir,
                "--attn-implementation", args.delta_attn_implementation,
            ]
        if args.method == "temp_lora":
            method_args += [*model_loading, "--seed", str(args.seed)]
        return module, [
            *method_args, "--query-style", args.query_style,
            "--max-new-tokens", str(args.max_new_tokens), *common,
        ]
    method_args = [
        "--method", args.method, "--model-label", args.model_label,
        "--output", str(output), "--max-new-tokens", str(args.max_new_tokens),
        "--query-style", args.query_style, "--oom-policy", "fail",
        *(
            metis_loading
            if args.method == "metis"
            else temp_lora_loading
            if args.method == "temp_lora"
            else model_loading
        ),
        *common,
    ]
    if args.method == "temp_lora":
        method_args += ["--seed", str(args.seed)]
    if args.model:
        method_args += ["--model-path", args.model]
    if args.checkpoint:
        method_args += ["--checkpoint", args.checkpoint]
    if args.adapter_dir:
        method_args += ["--adapter-dir", args.adapter_dir]
    return "eval.benchmarks.memops.scripts.run_memop_memory_baseline", method_args


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    if args.max_new_tokens is None:
        args.max_new_tokens = expected_max_new_tokens(
            config, args.method, args.benchmark
        )
    if args.max_input_tokens is None:
        args.max_input_tokens = expected_max_input_tokens(
            config, args.method, args.benchmark
        )
    if not args.instances:
        if not args.data_dir:
            raise ValueError("Pass --instances or --data-dir")
        args.instances = args.data_dir / f"{args.benchmark}.jsonl"
    if not args.dry_run:
        if not args.instances.is_file():
            raise FileNotFoundError(args.instances)
        if args.stage in {"inference", "all"}:
            if args.method in {"no_context", "full_context", "partial_context", "temp_lora", "delta_mem"} and not args.model:
                raise ValueError(f"--model is required for {args.method}")
            if args.method == "metis" and not args.checkpoint:
                raise ValueError("--checkpoint is required for metis")
            if args.method == "delta_mem" and not args.adapter_dir:
                raise ValueError("--adapter-dir is required for delta_mem")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module, cell_args = module_and_args(args, args.instances)
    inference_command = [sys.executable, "-m", module, *cell_args]
    raw = raw_output_path(args)
    score_command = [
        sys.executable,
        "-m",
        "eval.benchmarks.memqa.scripts.score_memqa",
        "--instances",
        str(args.instances),
        "--input",
        str(raw),
        "--output",
        str(args.output_dir / f"{args.benchmark}.{args.method}.scored.jsonl"),
        "--judge-base-url",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
        "--api-key-env",
        args.api_key_env,
        "--judge-repeats",
        "3",
        "--strict-only",
        "--strict-judge",
        "--fail-on-judge-error",
        "--concurrency",
        str(args.judge_concurrency),
        "--judge-max-attempts",
        str(args.judge_max_attempts),
        "--judge-retry-sleep",
        str(args.judge_retry_sleep),
        "--judge-retry-backoff",
        str(args.judge_retry_backoff),
        "--progress-every",
        str(args.judge_progress_every),
    ]
    commands = []
    if args.stage in {"inference", "all"}:
        commands.append(inference_command)
    if args.stage in {"score", "all"}:
        commands.append(score_command)
    manifest = {
        "experiment": config.get("experiment", "tables_5_6_11_main"),
        "benchmark": args.benchmark,
        "method": args.method,
        "stage": args.stage,
        "commands": commands,
        "paper_config": config,
    }
    write_json(args.output_dir / "cell_manifest.json", manifest)
    if not args.dry_run:
        for command in commands:
            env = None
            if args.method == "temp_lora":
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = str(args.seed)
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
