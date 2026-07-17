#!/usr/bin/env python3
"""Run the shared delta-mem method on normalized MemQA JSONL instances."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from eval.methods.delta_mem.delta_mem_baseline import DeltaMemBaseline, DeltaMemRuntimeConfig
from eval.methods.shared.memqa_io import audit_query_payload, output_record, read_jsonl, utc_now


def _run_instance(baseline: DeltaMemBaseline, instance: dict[str, Any]) -> tuple[list[dict[str, Any]], QueryResultLike, list[str], dict[str, Any]]:
    committed = [baseline.write(step) for step in instance.get("memory_steps", [])]
    result = baseline.query(instance["question"])
    audit_issues = audit_query_payload(instance, result.debug.get("query_payload", ""))
    return committed, result, audit_issues, {"query_context_allowed": False}


# Structural protocol alias; kept local to avoid importing typing_extensions.
QueryResultLike = Any


def run(args: argparse.Namespace) -> Path:
    instances = read_jsonl(args.instances, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_label = instances[0].get("dataset", "memqa") if instances else "memqa"
    config = DeltaMemRuntimeConfig(
        model_path=args.model_path,
        adapter_dir=args.adapter_dir,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        write_format=args.write_format,
        query_style=args.query_style,
    )
    baseline = DeltaMemBaseline(config)
    context_label = "delta_mem_memory_state_only"
    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memqa",
        "dataset": dataset_label,
        "baseline": "delta_mem",
        "model_label": args.model_label,
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "device": args.device,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "instances": str(args.instances),
        "instance_count": len(instances),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "write_format": args.write_format,
        "query_style": args.query_style,
        "context_policy": context_label,
        "context_policy_detail": "Clear messages/KV after every write; query with the question only.",
        "reset_policy": "DeltaMemChatSession.reset() before every MemQA record.",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            started = time.time()
            baseline.reset()
            committed, result, audit_issues, policy_debug = _run_instance(baseline, instance)
            if audit_issues and args.fail_on_audit_issue:
                raise RuntimeError(f"Query leakage audit failed for {instance['instance_id']}: {audit_issues}")
            record = output_record(
                run_id=args.run_id,
                baseline="delta_mem",
                model_label=args.model_label,
                model_path=args.model_path,
                instance=instance,
                instance_index=index,
                instance_count=len(instances),
                raw_output=result.raw_output,
                prompt_tokens=result.prompt_tokens,
                latency_sec=round(time.time() - started, 3),
                context_policy=context_label,
                generation_config={"do_sample": False, "max_new_tokens": args.max_new_tokens},
                extra={
                    "adapter_dir": args.adapter_dir,
                    "write_format": args.write_format,
                    "query_style": args.query_style,
                    "committed_steps": committed,
                    "committed_step_count": len(committed),
                    "query_latency_sec": result.latency_sec,
                    "delta_mem_debug": result.debug if args.include_debug else None,
                    "policy_debug": policy_debug,
                    "audit_issues": audit_issues,
                },
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "baseline": "delta_mem",
                        "context_policy": context_label,
                        "index": index,
                        "total": len(instances),
                        "instance_id": instance["instance_id"],
                        "audit_issues": len(audit_issues),
                        "latency_sec": record["latency_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local Qwen/Qwen3-4B-Instruct-2507 path or HF id if the official runtime can fetch it.")
    parser.add_argument("--adapter-dir", required=True, help="Local declare-lab/delta-mem_qwen3_4b-instruct adapter path or HF id.")
    parser.add_argument("--model-label", default="delta_mem_qwen3_4b_instruct_tsw")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="memqa_delta_mem")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--write-format", default="raw_memory_step", choices=["raw_memory_step", "instruction_wrapped"])
    parser.add_argument("--query-style", default="default", choices=["default", "memory_direct", "minimal"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--fail-on-audit-issue", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
