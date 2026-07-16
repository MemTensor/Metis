#!/usr/bin/env python
"""Run single- or multi-turn inference with a trained Metis checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from metis.checkpoint_utils import (
    is_delta_checkpoint,
    load_metis_model_from_checkpoint,
    resolve_backbone_path,
)
from metis.memory_utils import encode_and_commit_memory


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def apply_chat_template(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode_messages(
    tokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    *,
    add_generation_prompt: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    input_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    return input_ids, torch.ones_like(input_ids, device=device)


def tokenizer_source(args: argparse.Namespace, checkpoint: Path) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path
    if (checkpoint / "tokenizer_config.json").is_file():
        return str(checkpoint)
    if args.model_path:
        return args.model_path
    if is_delta_checkpoint(checkpoint):
        return resolve_backbone_path(checkpoint)
    return str(checkpoint)


def load_model_and_tokenizer(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)
    print(f"Loading checkpoint: {checkpoint}")
    print(f"Device: {device}; dtype: {dtype}")

    model = load_metis_model_from_checkpoint(
        checkpoint,
        model_path=args.model_path,
        backbone_type=args.backbone_type,
        device=device,
        dtype=dtype,
    )
    model.eval()

    source = tokenizer_source(args, checkpoint)
    print(f"Loading tokenizer: {source}")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: total={total_params:,}; trainable={trainable_params:,}")
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
    prompt: str,
) -> str:
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    input_ids, attention_mask = encode_messages(
        tokenizer,
        messages,
        device,
        add_generation_prompt=True,
    )

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.do_sample:
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
        })

    output = model.generate(**generation_kwargs)
    generated_ids = output[0, input_ids.size(1):].tolist()
    if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def commit_turn(
    model,
    tokenizer,
    device: torch.device,
    *,
    prompt: str,
    response: str,
    commit_mode: str,
) -> None:
    if commit_mode == "none":
        return
    messages = [{"role": "user", "content": prompt}]
    if commit_mode == "exchange":
        messages.append({"role": "assistant", "content": response})
    input_ids, attention_mask = encode_messages(
        tokenizer,
        messages,
        device,
        add_generation_prompt=False,
    )
    encode_and_commit_memory(model, input_ids, attention_mask=attention_mask)


def resolved_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or ["Hello, how are you?"]
    if len(prompts) == 1 and args.num_steps > 1:
        return prompts * args.num_steps
    if len(prompts) > 1 and args.num_steps not in (1, len(prompts)):
        raise ValueError(
            "With multiple --prompt values, --num_steps must be 1 or match "
            f"the number of prompts ({len(prompts)})."
        )
    return prompts


@torch.no_grad()
def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args)
    model.reset()

    rows: list[dict[str, Any]] = []
    for step, prompt in enumerate(resolved_prompts(args), start=1):
        response = generate_response(model, tokenizer, device, args, prompt)
        commit_turn(
            model,
            tokenizer,
            device,
            prompt=prompt,
            response=response,
            commit_mode=args.commit_mode,
        )
        row = {
            "step": step,
            "prompt": prompt,
            "response": response,
            "commit_mode": args.commit_mode,
        }
        rows.append(row)
        print(f"\n[Step {step}]")
        print(f"User: {prompt}")
        print(f"Assistant: {response}")
        print(f"Memory commit: {args.commit_mode}")

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved transcript: {output_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metis checkpoint inference with optional multi-turn memory commits",
    )
    parser.add_argument(
        "--checkpoint_path",
        "--ckpt",
        required=True,
        help="Metis delta or legacy full checkpoint directory",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional backbone override, useful after moving a delta checkpoint",
    )
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--backbone_type", choices=["qwen3_5", "qwen3", "llama"], default=None)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="User prompt; repeat this option for a multi-turn conversation",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Repeat a single prompt this many times (default: 1)",
    )
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--commit_mode", choices=["none", "user", "exchange"], default="exchange")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_jsonl", default=None)
    args = parser.parse_args()
    if args.num_steps < 1:
        parser.error("--num_steps must be >= 1")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
