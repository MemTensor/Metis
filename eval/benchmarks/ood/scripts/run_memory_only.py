#!/usr/bin/env python3
"""Resumable lane-owned runner for approved MemQA-OOD memory-only methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


from eval.methods.shared.memqa_io import (
    audit_query_payload,
    output_record,
    query_prompt_for_style,
    read_jsonl,
    utc_now,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_completed(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0
    completed: set[str] = set()
    lines = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            lines += 1
            row = json.loads(line)
            instance_id = str(row["instance_id"])
            if instance_id in completed:
                raise ValueError(f"duplicate existing instance_id at line {line_number}: {instance_id}")
            completed.add(instance_id)
    return completed, lines


def strict_query_audit(instance: dict[str, Any], query_payload: str) -> list[str]:
    issues = audit_query_payload(instance, query_payload)
    if instance.get("dataset") == "memdaily_official":
        # MemDaily's official evidence ids are small integer message offsets;
        # matching e.g. "0" inside an official time/choice is not leakage.
        issues = [issue for issue in issues if "forbidden evidence_id" not in issue]
    for step in instance.get("memory_steps", []):
        content = str(step.get("content", "")).strip()
        for fragment in (content[:160], content[-160:]):
            if len(fragment) >= 40 and fragment in query_payload:
                issues.append(f"query payload replays memory content: {fragment[:80]!r}")
    lowered = query_payload.lower()
    for key in ("supporting_evidence", "target_step_id", "ground_truth", "answer_text"):
        if key in lowered:
            issues.append(f"query payload contains scoring-only metadata key: {key}")
    return sorted(set(issues))


def build_config(args: argparse.Namespace, instances_sha256: str, count: int) -> dict[str, Any]:
    common = {
        "protocol": "memqa-ood-gold-v1-20260713",
        "run_id": args.run_id,
        "method": args.method,
        "model_label": args.model_label,
        "model_path": args.model_path,
        "instances": str(args.instances.resolve()),
        "instances_sha256": instances_sha256,
        "instance_count": count,
        "device": args.device,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "query_style": args.query_style,
        "context_policy": "memory_only; write memory_steps; query payload contains only frozen benchmark probe",
        "reset_policy": "reset before every instance",
        "resume_policy": "append-only; skip completed instance_id after config-digest verification",
        "fail_on_query_audit": True,
    }
    if args.method == "metis":
        common.update(
            {
                "checkpoint": args.model_path,
                "device_map": args.device_map,
                "model_parallel_devices": args.model_parallel_devices,
            }
        )
    elif args.method == "delta_mem":
        common.update(
            {
                "adapter_dir": args.adapter_dir,
                "write_format": args.write_format,
            }
        )
    else:
        common.update(
            {
                "write_format": args.write_format,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "train_epochs": args.train_epochs,
                "max_train_tokens": args.max_train_tokens,
                "write_chunk_tokens": args.write_chunk_tokens,
                "target_modules": args.target_modules,
                "gradient_checkpointing": args.gradient_checkpointing,
                "fail_on_write_truncation": True,
                "device_map": args.device_map,
                "max_memory": args.max_memory,
                "seed": args.seed,
            }
        )
    common["config_digest"] = stable_digest(common)
    return common


def prepare_output(args: argparse.Namespace, run_config: dict[str, Any]) -> tuple[set[str], Any]:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".meta.json")
    completed, line_count = load_completed(args.output)
    if completed:
        if not args.resume:
            raise ValueError(f"output exists and --no-resume was requested: {args.output}")
        if not meta_path.exists():
            raise ValueError(f"cannot resume without metadata: {meta_path}")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("run_config", {}).get("config_digest") != run_config["config_digest"]:
            raise ValueError("resume config digest mismatch")
    elif args.output.exists() and line_count == 0 and not args.resume:
        raise ValueError(f"empty output exists and --no-resume was requested: {args.output}")
    meta = {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "task": "memqa_ood",
        "run_config": run_config,
        "resume_state": {"completed_before_start": len(completed), "output_lines_before_start": line_count},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return completed, meta_path


def load_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.method == "delta_mem":
        from eval.methods.delta_mem.delta_mem_baseline import DeltaMemBaseline, DeltaMemRuntimeConfig

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
        runtime = DeltaMemBaseline(config)
        return runtime, {"adapter_dir": args.adapter_dir}
    if args.method == "temp_lora":
        from eval.methods.temp_lora.temp_lora_baseline import TempLoraBaseline, TempLoraRuntimeConfig

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
        runtime = TempLoraBaseline(config)
        return runtime, {"runtime_summary": runtime.runtime_summary()}

    from eval.methods.shared.metis_loader import (
        infer_input_device,
        load_v2_full_checkpoint,
        parse_dtype,
        parse_max_memory,
    )

    checkpoint = Path(args.model_path).expanduser().resolve()
    model, tokenizer, load_report = load_v2_full_checkpoint(
        checkpoint,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        device_map=args.device_map,
        model_parallel_devices=args.model_parallel_devices,
        max_memory=parse_max_memory(args.max_memory),
    )
    important_missing = load_report.get("important_missing", [])
    if important_missing:
        raise RuntimeError(f"Metis load_report has important_missing: {important_missing}")
    return (model, tokenizer, infer_input_device(model, args.device)), {"load_report": load_report}


def run_instance(args: argparse.Namespace, runtime: Any, instance: dict[str, Any]) -> tuple[str, int, float, dict[str, Any]]:
    if args.method == "metis":
        from eval.benchmarks.memqa.scripts.run_metis_memqa import commit_memory_steps, generate_answer

        model, tokenizer, input_device = runtime
        model.reset_memory()
        committed = commit_memory_steps(model, tokenizer, instance.get("memory_steps", []), input_device, _NoLowRank())
        raw_output, prompt_tokens, query_latency, query_payload = generate_answer(
            model,
            tokenizer,
            instance["question"],
            input_device,
            args.max_new_tokens,
            args.query_style,
        )
        issues = strict_query_audit(instance, query_payload)
        return raw_output, prompt_tokens, query_latency, {
            "committed_steps": committed,
            "committed_step_count": len(committed),
            "committed_tokens": sum(item["tokens"] for item in committed),
            "query_payload_sha256": hashlib.sha256(query_payload.encode()).hexdigest(),
            "audit_issues": issues,
        }

    runtime.reset()
    committed = [runtime.write(step) for step in instance.get("memory_steps", [])]
    if args.method == "temp_lora":
        truncated = sum(int(item.get("truncated_tokens", 0)) for item in committed)
        if truncated:
            raise RuntimeError(f"Temp-LoRA write truncation for {instance['instance_id']}: {truncated} tokens")
    result = runtime.query(instance["question"])
    query_payload = str(result.debug.get("query_payload", query_prompt_for_style(instance["question"], args.query_style)))
    issues = strict_query_audit(instance, query_payload)
    return result.raw_output, result.prompt_tokens, result.latency_sec, {
        "committed_steps": committed,
        "committed_step_count": len(committed),
        "query_payload_sha256": hashlib.sha256(query_payload.encode()).hexdigest(),
        "audit_issues": issues,
        "method_debug": result.debug if args.include_debug else None,
    }


class _NoLowRank:
    enabled = False

    def after_commit(self, step_id: Any = None) -> None:
        return None


def run(args: argparse.Namespace) -> None:
    instances = read_jsonl(args.instances, limit=args.limit)
    source_sha256 = sha256_file(args.instances)
    config = build_config(args, source_sha256, len(instances))
    completed, meta_path = prepare_output(args, config)
    pending = [row for row in instances if row["instance_id"] not in completed]
    if not pending:
        print(json.dumps({"event": "already_complete", "rows": len(completed)}, ensure_ascii=False))
        return
    runtime, runtime_meta = load_runtime(args)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["runtime"] = runtime_meta
    meta["updated_at"] = utc_now()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ordinal = {row["instance_id"]: index for index, row in enumerate(instances, 1)}
    with args.output.open("a", encoding="utf-8") as handle:
        for instance in pending:
            started = time.time()
            raw_output, prompt_tokens, query_latency, extra = run_instance(args, runtime, instance)
            if extra["audit_issues"]:
                raise RuntimeError(f"query audit failed for {instance['instance_id']}: {extra['audit_issues']}")
            record = output_record(
                run_id=args.run_id,
                baseline=args.method,
                model_label=args.model_label,
                model_path=args.model_path,
                instance=instance,
                instance_index=ordinal[instance["instance_id"]],
                instance_count=len(instances),
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                latency_sec=round(time.time() - started, 3),
                context_policy=f"{args.method}_memory_only_gold_evidence",
                generation_config={"do_sample": False, "max_new_tokens": args.max_new_tokens},
                extra={**extra, "query_latency_sec": query_latency},
            )
            record["task"] = "memqa_ood"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "event": "instance_complete",
                        "method": args.method,
                        "index": record["instance_index"],
                        "total": len(instances),
                        "instance_id": instance["instance_id"],
                        "latency_sec": record["latency_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["updated_at"] = utc_now()
    meta["resume_state"] = {
        "status": "complete",
        "completed_rows": len(load_completed(args.output)[0]),
        "expected_rows": len(instances),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("metis", "delta_mem", "temp_lora"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--query-style", default="minimal", choices=("default", "memory_direct", "minimal"))
    parser.add_argument("--write-format", default="raw_memory_step", choices=("raw_memory_step", "instruction_wrapped"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--device-map", default="single", choices=("single", "paired_layers", "auto", "balanced"))
    parser.add_argument("--model-parallel-devices", default="")
    parser.add_argument("--max-memory", action="append", default=None)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--max-train-tokens", type=int, default=4096)
    parser.add_argument("--write-chunk-tokens", type=int, default=1024)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,kv_proj,out_proj")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.method == "delta_mem" and not args.adapter_dir:
        parser.error("--adapter-dir is required for delta_mem")
    run(args)


if __name__ == "__main__":
    main()
