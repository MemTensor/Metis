#!/usr/bin/env python3
"""Run the Temp-LoRA adaptation baseline on normalized MemQA JSONL instances."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from eval.methods.shared.memqa_io import audit_query_payload, output_record, read_jsonl, utc_now
from eval.methods.temp_lora.temp_lora_baseline import TempLoraBaseline, TempLoraRuntimeConfig


def _run_instance(
    baseline: TempLoraBaseline,
    instance: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any, list[str], dict[str, Any]]:
    committed = [baseline.write(step) for step in instance.get("memory_steps", [])]
    result = baseline.query(instance["question"])
    audit_issues = audit_query_payload(instance, result.debug.get("query_payload", ""))
    return committed, result, audit_issues, {"query_context_allowed": False}


def run(args: argparse.Namespace) -> Path:
    instances = read_jsonl(args.instances, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_label = instances[0].get("dataset", "memqa") if instances else "memqa"
    config = TempLoraRuntimeConfig(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_epochs=args.train_epochs,
        max_train_tokens=args.max_train_tokens,
        write_chunk_tokens=args.write_chunk_tokens,
        target_modules=args.target_modules,
        write_format=args.write_format,
        query_style=args.query_style,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map=args.device_map,
        max_memory=tuple(args.max_memory or ()),
        seed=args.seed,
    )
    baseline = TempLoraBaseline(config)
    context_label = "temp_lora_memory_state_only"
    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memqa",
        "dataset": dataset_label,
        "baseline": "temp_lora",
        "model_label": args.model_label,
        "model_path": args.model_path,
        "device": args.device,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "instances": str(args.instances),
        "instance_count": len(instances),
        "context_policy": context_label,
        "context_policy_detail": "Train a temporary LoRA on memory_steps; query with the question only.",
        "reset_policy": "Reset temporary LoRA weights and optimizer before every MemQA record.",
        "query_style": args.query_style,
        "max_train_tokens": args.max_train_tokens,
        "write_chunk_tokens": args.write_chunk_tokens,
        "truncation_audit": "Each committed step records raw_write_tokens, write_tokens, and truncated_tokens.",
        "runtime_summary": baseline.runtime_summary(),
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
                baseline="temp_lora",
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
                    "write_format": args.write_format,
                    "query_style": args.query_style,
                    "max_train_tokens": args.max_train_tokens,
                    "write_chunk_tokens": args.write_chunk_tokens,
                    "committed_steps": committed,
                    "committed_step_count": len(committed),
                    "query_latency_sec": result.latency_sec,
                    "temp_lora_debug": result.debug if args.include_debug else None,
                    "policy_debug": policy_debug,
                    "audit_issues": audit_issues,
                },
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "baseline": "temp_lora",
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
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", default="temp_lora_official_like_memory_direct")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="memqa_temp_lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--max-train-tokens", type=int, default=4096)
    parser.add_argument("--write-chunk-tokens", type=int, default=1024)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,kv_proj,out_proj")
    parser.add_argument("--write-format", default="raw_memory_step", choices=["raw_memory_step", "instruction_wrapped"])
    parser.add_argument("--query-style", default="memory_direct", choices=["default", "memory_direct", "minimal"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="single", choices=["single", "auto", "balanced"])
    parser.add_argument("--max-memory", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--fail-on-audit-issue", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
