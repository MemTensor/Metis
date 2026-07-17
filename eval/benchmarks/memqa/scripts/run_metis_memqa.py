#!/usr/bin/env python3
"""Run Metis checkpoints on normalized MemQA instances."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


from eval.methods.shared.memqa_io import audit_query_payload, query_prompt_for_style  # noqa: E402

from eval.methods.shared.metis_loader import (
    _chat_template,
    infer_input_device,
    load_v2_full_checkpoint,
    parse_dtype,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_messages(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool, device: str) -> torch.Tensor:
    text = _chat_template(tokenizer, messages, add_generation_prompt=add_generation_prompt)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long, device=device)


def memory_message(step: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Conversation memory segment.",
            "Commit the following dated dialogue segment to memory for later question answering.",
            "",
            str(step.get("content", "")),
        ]
    )


@torch.no_grad()
def commit_memory_steps(
    model: Any,
    tokenizer: Any,
    steps: list[dict[str, Any]],
    device: str,
    low_rank: MetisLowRankLocalMemoryProjector,
) -> list[dict[str, Any]]:
    committed = []
    for step in steps:
        messages = [{"role": "user", "content": memory_message(step)}]
        input_ids = render_messages(tokenizer, messages, add_generation_prompt=False, device=device)
        started = time.time()
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=None,
            commit_memory=True,
            use_cache=False,
        )
        projection_event = low_rank.after_commit(step_id=step.get("step_id"))
        committed.append(
            {
                "step_id": step.get("step_id"),
                "turn_start": step.get("turn_start"),
                "turn_end": step.get("turn_end"),
                "tokens": int(input_ids.shape[1]),
                "latency_sec": round(time.time() - started, 3),
                "low_rank_projection": projection_event["summary"] if projection_event else None,
            }
        )
    return committed


@torch.no_grad()
def generate_answer(model: Any, tokenizer: Any, question: str, device: str, max_new_tokens: int, query_style: str) -> tuple[str, int, float, str]:
    prompt = query_prompt_for_style(question, query_style)
    started = time.time()
    prefix_ids = render_messages(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True, device=device)
    output_ids = model.generate(
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[:, prefix_ids.shape[1] :]
    generated = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    return generated, int(prefix_ids.shape[1]), round(time.time() - started, 3), prompt


def config_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"checkpoint_dir": str(path)}
    config_path = path / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ["architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads"]:
            out[key] = cfg.get(key)
    trainer_state = path / "trainer_state.json"
    if trainer_state.exists():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        out["trainer_global_step"] = state.get("global_step")
        out["trainer_max_steps"] = state.get("max_steps")
    return out


def run(args: argparse.Namespace) -> Path:
    instances = read_jsonl(args.instances)
    if args.limit:
        instances = instances[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset_label = instances[0].get("dataset", "memqa") if instances else "memqa"

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    model, tokenizer, load_report = load_v2_full_checkpoint(
        ckpt_path,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        device_map=args.device_map,
        model_parallel_devices=args.model_parallel_devices,
        max_memory=parse_max_memory(args.max_memory),
    )
    input_device = infer_input_device(model, args.device)
    low_rank_config = build_low_rank_local_memory_config(
        enabled=args.metis_low_rank_local_memory,
        rank=args.metis_low_rank_rank,
        policy=args.metis_low_rank_policy,
        target=args.metis_low_rank_target,
    )
    low_rank = MetisLowRankLocalMemoryProjector(model, low_rank_config)

    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memqa",
        "dataset": dataset_label,
        "baseline": "metis",
        "model_label": args.model_label,
        "model_path": str(ckpt_path),
        "checkpoint_config": config_summary(ckpt_path),
        "load_report": load_report,
        "device": args.device,
        "input_device": str(input_device),
        "device_map": args.device_map,
        "model_parallel_devices": args.model_parallel_devices,
        "max_memory": args.max_memory,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "instances": str(args.instances),
        "instance_count": len(instances),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "context_policy": "memory_steps committed with commit_memory=True; query sees only the question prompt.",
        "query_style": args.query_style,
        "reset_policy": "model.reset_memory() before every instance.",
        "low_rank_local_memory": low_rank.config_dict(),
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            started = time.time()
            model.reset_memory()
            low_rank.reset_record_stats()
            committed = commit_memory_steps(
                model,
                tokenizer,
                instance.get("memory_steps", []),
                input_device,
                low_rank,
            )
            low_rank.before_query()
            raw_output, prompt_tokens, query_latency, query_payload = generate_answer(
                model,
                tokenizer,
                instance["question"],
                input_device,
                args.max_new_tokens,
                args.query_style,
            )
            audit_issues = audit_query_payload(instance, query_payload)
            if audit_issues and args.fail_on_audit_issue:
                raise RuntimeError(f"Query leakage audit failed for {instance['instance_id']}: {audit_issues}")
            record = {
                "run_id": args.run_id,
                "date": utc_now(),
                "task": "memqa",
                "dataset": instance.get("dataset"),
                "split": instance.get("split"),
                "baseline": "metis",
                "model_label": args.model_label,
                "model_path": str(ckpt_path),
                "device": args.device,
                "input_device": str(input_device),
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "instance_index": index,
                "instance_count": len(instances),
                "instance_id": instance["instance_id"],
                "source_sample_id": instance.get("source_sample_id"),
                "raw_category": instance.get("metadata", {}).get("raw_category"),
                "is_adversarial": instance.get("metadata", {}).get("is_adversarial"),
                "question": instance["question"],
                "context_policy": "metis_memory_only",
                "committed_steps": committed,
                "committed_step_count": len(committed),
                "committed_tokens": sum(item["tokens"] for item in committed),
                "prompt_tokens": prompt_tokens,
                "latency_sec": round(time.time() - started, 3),
                "query_latency_sec": query_latency,
                "raw_output": raw_output,
                "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "query_style": args.query_style,
                "audit_issues": audit_issues,
                "method_debug": {"low_rank_local_memory": low_rank.record_debug()} if low_rank.enabled else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "baseline": "metis",
                        "model_label": args.model_label,
                        "index": index,
                        "total": len(instances),
                        "instance_id": instance["instance_id"],
                        "latency_sec": record["latency_sec"],
                    },
                    ensure_ascii=False,
                )
            )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="memqa_metis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="single", choices=["single", "paired_layers", "auto"])
    parser.add_argument("--model-parallel-devices", default="")
    parser.add_argument("--max-memory", action="append", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--query-style", default="default", choices=["default", "memory_direct", "minimal"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-audit-issue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metis-low-rank-local-memory", action="store_true")
    parser.add_argument("--metis-low-rank-rank", type=int, default=None)
    parser.add_argument(
        "--metis-low-rank-policy",
        default="after_each_commit",
        choices=sorted(LOW_RANK_POLICIES),
    )
    parser.add_argument("--metis-low-rank-target", default="state", choices=sorted(LOW_RANK_TARGETS))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
