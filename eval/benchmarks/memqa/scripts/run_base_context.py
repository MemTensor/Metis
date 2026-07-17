#!/usr/bin/env python3
"""Run base-model MemQA no-context and full-context baselines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(" ", "_")


def _format_dialog_turn(turn: dict[str, Any]) -> str:
    line = f"{turn.get('dia_id')} {turn.get('speaker')} said: \"{turn.get('text', '')}\""
    if turn.get("blip_caption"):
        line += f" Shared image caption: {turn['blip_caption']}"
    return line


def context_text(instance: dict[str, Any]) -> str:
    context = instance.get("context", [])
    if any("ref_id" in item for item in context):
        lines = ["Reference context:"]
        for item in context:
            lines.append(f"{item.get('ref_id')}: {item.get('text', '')}")
        return "\n".join(lines)
    memory_steps = instance.get("memory_steps", [])
    if instance.get("dataset") == "nextmem_stm" and memory_steps:
        lines = ["Reference context:"]
        for step in memory_steps:
            content = str(step.get("content", ""))
            if content.startswith("REFERENCE:\n"):
                content = content.removeprefix("REFERENCE:\n")
            lines.append(content)
        return "\n".join(lines)

    if instance.get("split") == "evidence_sessions" or any("session_id" in turn for turn in context):
        lines = ["Evidence-session context:"]
        sessions = {item.get("session_id"): item for item in instance.get("sessions", []) if item.get("session_id")}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        order: list[str] = []
        for turn in context:
            session_id = str(turn.get("session_id") or "session_unknown")
            if session_id not in grouped:
                order.append(session_id)
            grouped[session_id].append(turn)
        for session_id in order:
            session = sessions.get(session_id, {})
            date_time = session.get("date_time") or grouped[session_id][0].get("date_time")
            lines.append(f"SESSION: {session_id}")
            if date_time:
                lines.append(f"DATE: {date_time}")
            for turn in grouped[session_id]:
                lines.append(_format_dialog_turn(turn))
        return "\n".join(lines)

    lines = ["Single-session context:"]
    session = instance.get("session", {})
    if session.get("date_time"):
        lines.append(f"DATE: {session['date_time']}")
    for turn in context:
        lines.append(_format_dialog_turn(turn))
    return "\n".join(lines)


def prompt_for(instance: dict[str, Any], condition: str) -> str:
    question = instance["question"]
    qa = f"Question: {question}\nAnswer with a short phrase. If the answer is not known from the given information, say \"No information available\"."
    if condition == "no_context":
        return qa
    if condition == "full_context":
        return context_text(instance) + "\n\n" + qa
    raise ValueError(f"Unsupported condition: {condition}")


def chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.no_grad()
def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, int, float]:
    started = time.time()
    text = chat_text(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[:, encoded.input_ids.shape[1] :]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip(), int(encoded.input_ids.shape[1]), round(time.time() - started, 3)


def parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    out: dict[int | str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --max-memory item {item!r}; expected DEVICE:VALUE, e.g. 0:75GiB")
        key, value = item.split(":", 1)
        out[int(key) if key.isdigit() else key] = value
    return out


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> Any:
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "single":
        kwargs["device_map"] = {"": args.device}
    elif args.device_map == "auto":
        kwargs["device_map"] = "auto"
        max_memory = parse_max_memory(args.max_memory)
        if max_memory:
            kwargs["max_memory"] = max_memory
    else:
        raise ValueError(f"Unsupported device_map: {args.device_map}")
    return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)


def run_condition(args: argparse.Namespace, model: Any, tokenizer: Any, instances: list[dict[str, Any]], condition: str) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{safe_label(args.model_label)}.{condition}.raw.jsonl"
    dataset_label = instances[0].get("dataset", "memqa") if instances else "memqa"
    hf_device_map = getattr(model, "hf_device_map", None)
    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memqa",
        "dataset": dataset_label,
        "model_label": args.model_label,
        "model_path": args.model_path,
        "condition": condition,
        "condition_policy": "no_context asks only the question; full_context prepends formatted context/reference text.",
        "device": args.device,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "hf_device_map": {str(k): str(v) for k, v in hf_device_map.items()} if isinstance(hf_device_map, dict) else hf_device_map,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "instances": str(args.instances),
        "instance_count": len(instances),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "thinking": "disabled via tokenizer.apply_chat_template(enable_thinking=False) when supported",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            prompt = prompt_for(instance, condition)
            raw_output, prompt_tokens, latency_sec = generate(model, tokenizer, prompt, args.device, args.max_new_tokens)
            record = {
                "run_id": args.run_id,
                "date": utc_now(),
                "task": "memqa",
                "dataset": instance.get("dataset"),
                "split": instance.get("split"),
                "baseline": f"base_{condition}",
                "model_label": args.model_label,
                "model_path": args.model_path,
                "device": args.device,
                "device_map": args.device_map,
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "instance_index": index,
                "instance_count": len(instances),
                "instance_id": instance["instance_id"],
                "source_sample_id": instance.get("source_sample_id"),
                "source_dataset": instance.get("metadata", {}).get("source_dataset"),
                "raw_category": instance.get("metadata", {}).get("raw_category"),
                "is_adversarial": instance.get("metadata", {}).get("is_adversarial"),
                "question": instance["question"],
                "context_policy": condition,
                "prompt_tokens": prompt_tokens,
                "latency_sec": latency_sec,
                "raw_output": raw_output,
                "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"condition": condition, "index": index, "total": len(instances), "instance_id": instance["instance_id"], "latency_sec": latency_sec}, ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="memqa_base_context")
    parser.add_argument("--conditions", nargs="+", default=["no_context", "full_context"], choices=["no_context", "full_context"])
    parser.add_argument("--device", default="cuda:0", help="Input tensor device. For --device-map auto, use the first visible CUDA device.")
    parser.add_argument("--device-map", default="single", choices=["single", "auto"])
    parser.add_argument("--max-memory", nargs="*", default=None, help="Only used with --device-map auto; items like 0:75GiB 1:75GiB.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    instances = read_jsonl(args.instances)
    if args.limit:
        instances = instances[: args.limit]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args, dtype)
    model.eval()

    for condition in args.conditions:
        run_condition(args, model, tokenizer, instances, condition)


if __name__ == "__main__":
    main()
