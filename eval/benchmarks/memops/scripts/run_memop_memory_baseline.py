#!/usr/bin/env python3
"""Run memory-state baselines on normalized MemOP JSONL instances."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


from eval.methods.shared.memqa_io import query_prompt_for_style  # noqa: E402
from eval.methods.shared.memory_contract import QueryResult  # noqa: E402
from eval.methods.shared.metis_loader import (
    _chat_template as metis_chat_template,
    infer_input_device,
    load_v2_full_checkpoint,
    parse_max_memory,
)  # noqa: E402
from eval.methods.shared.metis_low_rank_memory import (  # noqa: E402
    LOW_RANK_POLICIES,
    LOW_RANK_TARGETS,
    MetisLowRankLocalMemoryProjector,
    build_low_rank_local_memory_config,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def _text_fragment(value: Any, max_len: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text[:max_len]


def _context_text(item: dict[str, Any]) -> str:
    if item.get("content") is not None:
        return str(item.get("content", ""))
    messages = item.get("messages")
    if isinstance(messages, list):
        lines = []
        for message in messages:
            role = str(message.get("role", "message")).upper()
            content = str(message.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    return str(item.get("text", ""))


def audit_query_payload(instance: dict[str, Any], query_text: str) -> list[str]:
    """Check that a memory-only query prompt did not replay forbidden material."""

    issues: list[str] = []
    fragments: list[tuple[str, str]] = []
    question = _text_fragment(instance.get("question"))
    for step in instance.get("memory_steps", []):
        fragment = _text_fragment(step.get("content"))
        if len(fragment) >= 40:
            fragments.append(("memory_step", fragment))
    for item in instance.get("context", []) or []:
        fragment = _text_fragment(_context_text(item))
        if len(fragment) >= 40:
            fragments.append(("context", fragment))
    answer = _text_fragment(instance.get("answer"))
    if len(answer) >= 20 and answer not in question:
        fragments.append(("gold_answer", answer))
    for evidence in instance.get("evidence", []) or []:
        evidence_text = _text_fragment(str(evidence))
        # MetisTest evidence IDs can be short names like "T0", which may also
        # be legitimate entities in the user question. Treat only high-signal
        # IDs outside the question as query-leakage evidence.
        if len(evidence_text) >= 8 and evidence_text not in question:
            fragments.append(("evidence_id", evidence_text))

    for label, fragment in fragments:
        if fragment and fragment in query_text and fragment not in question:
            issues.append(f"query payload includes forbidden {label}: {fragment[:80]!r}")
    if "rubric" in query_text.lower() or "gold answer" in query_text.lower():
        issues.append("query payload includes rubric/gold-answer wording")
    return issues


def memory_message(step: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Conversation memory segment.",
            "Commit the following dialogue segment to memory for later question answering.",
            "",
            str(step.get("content", "")),
        ]
    )


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or exc.__class__.__name__ == "OutOfMemoryError"


def clear_cuda_after_oom() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def oom_placeholder_result(question: str, query_style: str, exc: BaseException) -> QueryResult:
    return QueryResult(
        raw_output="由于 OOM 无法回答",
        prompt_tokens=0,
        latency_sec=0.0,
        debug={
            "query_payload": query_prompt_for_style(question, query_style),
            "runtime_status": "oom_placeholder",
            "runtime_error": f"{exc.__class__.__name__}: {str(exc).splitlines()[0]}",
        },
    )


def metis_parse_dtype(value: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[value]


def render_metis_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool, device: str) -> torch.Tensor:
    text = metis_chat_template(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long, device=device)


class MetisMemOPBaseline:
    method_id = "metis"

    def __init__(
        self,
        checkpoint: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        query_style: str,
        device_map: str = "single",
        model_parallel_devices: str = "",
        max_memory: list[str] | None = None,
        low_rank_enabled: bool = False,
        low_rank_rank: int | None = None,
        low_rank_policy: str = "after_each_commit",
        low_rank_target: str = "state",
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.query_style = query_style
        self.model, self.tokenizer, self.load_report = load_v2_full_checkpoint(
            self.checkpoint,
            device=device,
            dtype=metis_parse_dtype(dtype),
            device_map=device_map,
            model_parallel_devices=model_parallel_devices,
            max_memory=parse_max_memory(max_memory),
        )
        self.input_device = infer_input_device(self.model, device)
        self.low_rank_config = build_low_rank_local_memory_config(
            enabled=low_rank_enabled,
            rank=low_rank_rank,
            policy=low_rank_policy,
            target=low_rank_target,
        )
        self.low_rank = MetisLowRankLocalMemoryProjector(self.model, self.low_rank_config)

    def reset(self) -> None:
        self.model.reset_memory()
        self.low_rank.reset_record_stats()

    @torch.no_grad()
    def write(self, step: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": memory_message(step)}]
        input_ids = render_metis_messages(self.tokenizer, messages, add_generation_prompt=False, device=self.input_device)
        started = time.time()
        self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=None,
            commit_memory=True,
            use_cache=False,
        )
        projection_event = self.low_rank.after_commit(step_id=step.get("step_id"))
        return {
            "step_id": step.get("step_id"),
            "turn_start": step.get("turn_start"),
            "turn_end": step.get("turn_end"),
            "commit_granularity": step.get("commit_granularity"),
            "tokens": int(input_ids.shape[1]),
            "latency_sec": round(time.time() - started, 3),
            "low_rank_projection": projection_event["summary"] if projection_event else None,
        }

    @torch.no_grad()
    def query(self, question: str) -> QueryResult:
        prompt = query_prompt_for_style(question, self.query_style)
        started = time.time()
        self.low_rank.before_query()
        prefix_ids = render_metis_messages(
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            device=self.input_device,
        )
        output_ids = self.model.generate(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = output_ids[:, prefix_ids.shape[1] :]
        return QueryResult(
            raw_output=self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip(),
            prompt_tokens=int(prefix_ids.shape[1]),
            latency_sec=round(time.time() - started, 3),
            debug={
                "query_payload": prompt,
                "load_report": self.load_report,
                "input_device": str(self.input_device),
                "low_rank_local_memory": self.low_rank.record_debug() if self.low_rank.enabled else None,
            },
        )


def build_baseline(args: argparse.Namespace) -> tuple[Any, str, str]:
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
        return DeltaMemBaseline(config), args.model_path, "delta_mem_memory_state_only"

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
        return TempLoraBaseline(config), args.model_path, "temp_lora_memory_state_only"

    if args.method == "metis":
        return (
            MetisMemOPBaseline(
                args.checkpoint,
                args.device,
                args.dtype,
                args.max_new_tokens,
                args.query_style,
                args.device_map,
                args.model_parallel_devices,
                args.max_memory,
                args.metis_low_rank_local_memory,
                args.metis_low_rank_rank,
                args.metis_low_rank_policy,
                args.metis_low_rank_target,
            ),
            str(Path(args.checkpoint).expanduser().resolve()),
            "metis_memory_state_only",
        )

    raise ValueError(f"Unsupported method: {args.method}")


def output_record(
    *,
    args: argparse.Namespace,
    instance: dict[str, Any],
    index: int,
    total: int,
    model_path: str,
    context_policy: str,
    committed: list[dict[str, Any]],
    result: QueryResult,
    audit_issues: list[str],
    latency_sec: float,
    runtime_status: str = "ok",
    runtime_error: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "date": utc_now(),
        "task": "memop",
        "task_type": instance.get("task_type"),
        "dataset": instance.get("dataset"),
        "split": instance.get("split"),
        "setting": instance.get("setting"),
        "operation": instance.get("operation"),
        "subtask": instance.get("subtask"),
        "baseline": args.method,
        "model_label": args.model_label,
        "model_path": model_path,
        "adapter_dir": args.adapter_dir if args.method == "delta_mem" else None,
        "checkpoint": args.checkpoint if args.method == "metis" else None,
        "device": args.device,
        "input_device": getattr(args, "metis_input_device", args.device),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "instance_index": index,
        "instance_count": total,
        "instance_id": instance["instance_id"],
        "source_sample_id": instance.get("source_sample_id"),
        "question": instance["question"],
        "context_policy": context_policy,
        "query_style": args.query_style,
        "committed_steps": committed,
        "committed_step_count": len(committed),
        "prompt_tokens": result.prompt_tokens,
        "latency_sec": latency_sec,
        "query_latency_sec": result.latency_sec,
        "raw_output": result.raw_output,
        "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "audit_issues": audit_issues,
        "runtime_status": runtime_status,
        "runtime_error": runtime_error,
        "method_debug": result.debug if args.include_debug or args.metis_low_rank_local_memory else None,
    }


def run(args: argparse.Namespace) -> Path:
    instances = read_jsonl(args.instances, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline, model_path, context_policy = build_baseline(args)
    args.metis_input_device = str(getattr(baseline, "input_device", args.device))
    args.metis_load_report = getattr(baseline, "load_report", None)
    dataset_label = instances[0].get("dataset", "memop") if instances else "memop"
    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memop",
        "dataset": dataset_label,
        "method": args.method,
        "baseline": args.method,
        "model_label": args.model_label,
        "model_path": model_path,
        "adapter_dir": args.adapter_dir if args.method == "delta_mem" else None,
        "checkpoint": args.checkpoint if args.method == "metis" else None,
        "instances": str(args.instances),
        "instance_count": len(instances),
        "context_policy": context_policy,
        "query_policy": "memory_steps are consumed in the write phase; query receives question-only prompt plus method memory state.",
        "query_style": args.query_style,
        "reset_policy": "reset baseline state before every MemOP record.",
        "device": args.device,
        "input_device": getattr(args, "metis_input_device", args.device),
        "device_map": args.device_map if args.method in {"metis", "temp_lora"} else None,
        "model_parallel_devices": args.model_parallel_devices if args.method == "metis" else None,
        "max_memory": args.max_memory if args.method in {"metis", "temp_lora"} else None,
        "load_report": getattr(args, "metis_load_report", None) if args.method == "metis" else None,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "write_format": args.write_format,
        "oom_policy": args.oom_policy,
    }
    if args.method == "metis":
        meta["low_rank_local_memory"] = getattr(baseline, "low_rank_config", None).to_dict()
    if args.method == "temp_lora":
        meta["temp_lora"] = baseline.runtime_summary()
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            started = time.time()
            committed: list[dict[str, Any]] = []
            runtime_status = "ok"
            runtime_error = None
            try:
                baseline.reset()
                for step in instance.get("memory_steps", []):
                    committed.append(baseline.write(step))
                result = baseline.query(instance["question"])
                audit_issues = audit_query_payload(instance, result.debug.get("query_payload", ""))
                if audit_issues and args.fail_on_audit_issue:
                    raise RuntimeError(f"Query leakage audit failed for {instance['instance_id']}: {audit_issues}")
            except Exception as exc:
                if args.oom_policy != "placeholder" or not is_cuda_oom(exc):
                    raise
                clear_cuda_after_oom()
                runtime_status = "oom_placeholder"
                runtime_error = f"{exc.__class__.__name__}: {str(exc).splitlines()[0]}"
                result = oom_placeholder_result(instance["question"], args.query_style, exc)
                audit_issues = []
            record = output_record(
                args=args,
                instance=instance,
                index=index,
                total=len(instances),
                model_path=model_path,
                context_policy=context_policy,
                committed=committed,
                result=result,
                audit_issues=audit_issues,
                latency_sec=round(time.time() - started, 3),
                runtime_status=runtime_status,
                runtime_error=runtime_error,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "method": args.method,
                        "context_policy": context_policy,
                        "index": index,
                        "total": len(instances),
                        "instance_id": instance["instance_id"],
                        "audit_issues": len(audit_issues),
                        "runtime_status": runtime_status,
                        "latency_sec": record["latency_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["delta_mem", "temp_lora", "metis"])
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="memop_memory_baseline")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="single", choices=["single", "paired_layers", "auto", "balanced"])
    parser.add_argument("--model-parallel-devices", default="")
    parser.add_argument("--max-memory", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--write-format", default="raw_memory_step", choices=["raw_memory_step", "instruction_wrapped"])
    parser.add_argument("--query-style", default="memory_direct", choices=["default", "memory_direct", "minimal"])
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--fail-on-audit-issue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oom-policy", choices=["fail", "placeholder"], default="placeholder")
    parser.add_argument("--metis-low-rank-local-memory", action="store_true")
    parser.add_argument("--metis-low-rank-rank", type=int, default=None)
    parser.add_argument(
        "--metis-low-rank-policy",
        default="after_each_commit",
        choices=sorted(LOW_RANK_POLICIES),
    )
    parser.add_argument("--metis-low-rank-target", default="state", choices=sorted(LOW_RANK_TARGETS))
    args = parser.parse_args()

    if args.method in {"delta_mem", "temp_lora"} and not args.model_path:
        raise ValueError(f"--model-path is required for {args.method}")
    if args.method == "delta_mem" and not args.adapter_dir:
        raise ValueError("--adapter-dir is required for delta_mem")
    if args.method == "metis" and not args.checkpoint:
        raise ValueError("--checkpoint is required for metis")
    run(args)


if __name__ == "__main__":
    main()
