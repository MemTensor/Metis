#!/usr/bin/env python3
"""Check whether the Temp-LoRA adaptation runtime is importable."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from eval.methods.temp_lora.temp_lora_baseline import TempLoraBaseline, TempLoraRuntimeConfig


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-model", action="store_true", help="Actually load the model and attach a small LoRA.")
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--max-train-tokens", type=int, default=128)
    args = parser.parse_args()

    report = {
        "runtime_import": TempLoraBaseline.runtime_check(),
        "packages": {pkg: package_version(pkg) for pkg in ["torch", "transformers", "peft", "accelerate", "deepspeed", "datasets"]},
        "model_path": args.model_path,
        "model_path_is_local": Path(args.model_path).exists(),
        "load_model": args.load_model,
    }

    if args.load_model:
        baseline = TempLoraBaseline(
            TempLoraRuntimeConfig(
                model_path=args.model_path,
                device=args.device,
                dtype=args.dtype,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_rank,
                max_train_tokens=args.max_train_tokens,
                train_epochs=1,
            )
        )
        report["loaded_runtime"] = baseline.runtime_summary()
        baseline.reset()

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
