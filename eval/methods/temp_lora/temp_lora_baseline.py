"""Reusable Temp-LoRA adaptation for parameterized-memory baselines.

This wrapper adapts the official Temp-LoRA idea to the shared MemQA/MemOP
contract. The official paper trains a temporary LoRA module on previous chunks
during long-text generation. Here the information phase trains the temporary
LoRA on the instance memory stream; the query phase receives only the question
prompt plus the learned LoRA weights.

This is an adaptation baseline, not a claim that the official repository ships
a ready-made memory-QA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import Any

from eval.methods.shared.memqa_io import memory_step_prompt, query_prompt_for_style
from eval.methods.shared.memory_contract import QueryResult


_DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,kv_proj,out_proj"


@dataclass(frozen=True)
class TempLoraRuntimeConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_new_tokens: int = 96
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    train_epochs: int = 2
    max_train_tokens: int = 4096
    write_chunk_tokens: int = 1024
    min_train_tokens: int = 8
    target_modules: str = _DEFAULT_TARGET_MODULES
    write_format: str = "raw_memory_step"
    query_style: str = "memory_direct"
    gradient_checkpointing: bool = False
    device_map: str = "single"
    max_memory: tuple[str, ...] = ()
    seed: int = 20260702


class TempLoraRuntimeUnavailable(RuntimeError):
    pass


class TempLoraBaseline:
    method_id = "temp_lora"

    def __init__(self, config: TempLoraRuntimeConfig):
        self.config = config
        runtime = self._load_runtime()
        self._torch = runtime["torch"]
        self._nn = runtime["nn"]
        self._AutoModelForCausalLM = runtime["AutoModelForCausalLM"]
        self._AutoTokenizer = runtime["AutoTokenizer"]
        self._LoraConfig = runtime["LoraConfig"]
        self._TaskType = runtime["TaskType"]
        self._get_peft_model = runtime["get_peft_model"]
        self._transformers_set_seed = runtime["transformers_set_seed"]
        self._set_seed()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self._target_modules = self._resolve_target_modules(config.target_modules)
        self.model = self._attach_lora(self.model, self._target_modules)
        self.input_device = self._infer_input_device()
        self._initial_lora_state = self._snapshot_trainable_state()
        self.optimizer = self._build_optimizer()
        self._write_count = 0

    @staticmethod
    def _load_runtime() -> dict[str, Any]:
        try:
            import torch
            from torch import nn
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:  # pragma: no cover - depends on external runtime.
            raise TempLoraRuntimeUnavailable(
                "Temp-LoRA adaptation requires torch, transformers, and peft. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return {
            "torch": torch,
            "nn": nn,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "LoraConfig": LoraConfig,
            "TaskType": TaskType,
            "get_peft_model": get_peft_model,
            "transformers_set_seed": set_seed,
        }

    def _set_seed(self) -> None:
        seed = int(self.config.seed)
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(seed)
        self._transformers_set_seed(seed)

    @classmethod
    def runtime_check(cls) -> dict[str, Any]:
        try:
            cls._load_runtime()
            return {"available": True, "error": None}
        except TempLoraRuntimeUnavailable as exc:
            return {"available": False, "error": str(exc)}

    @property
    def target_modules(self) -> list[str]:
        return list(self._target_modules)

    def _torch_dtype(self) -> Any:
        mapping = {
            "bfloat16": self._torch.bfloat16,
            "float16": self._torch.float16,
            "float32": self._torch.float32,
        }
        try:
            return mapping[self.config.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {self.config.dtype}") from exc

    def _load_tokenizer(self) -> Any:
        tokenizer = self._AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "eod_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
        return tokenizer

    def _load_model(self) -> Any:
        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.model_path,
            "trust_remote_code": True,
            "torch_dtype": self._torch_dtype(),
        }
        if self.config.device_map == "single":
            kwargs["device_map"] = {"": self.config.device}
        elif self.config.device_map in {"auto", "balanced"}:
            kwargs["device_map"] = self.config.device_map
            kwargs["low_cpu_mem_usage"] = True
            max_memory = self._parse_max_memory(self.config.max_memory)
            if max_memory:
                kwargs["max_memory"] = max_memory
        else:
            raise ValueError(
                "Temp-LoRA device_map must be single, auto, or balanced; "
                f"got {self.config.device_map!r}"
            )
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        model = self._AutoModelForCausalLM.from_pretrained(**kwargs)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        return model

    @staticmethod
    def _parse_max_memory(items: tuple[str, ...]) -> dict[int, str] | None:
        result: dict[int, str] = {}
        for index, item in enumerate(items):
            key, separator, value = item.partition(":")
            if separator:
                result[int(key.strip())] = value.strip()
            else:
                result[index] = item.strip()
        return result or None

    def _infer_input_device(self) -> Any:
        try:
            embeddings = self.model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None:
                return weight.device
        except Exception:
            pass
        for parameter in self.model.parameters():
            return parameter.device
        return self._torch.device(self.config.device)

    def _linear_suffixes(self) -> set[str]:
        suffixes: set[str] = set()
        for name, module in self.model.named_modules():
            if isinstance(module, self._nn.Linear):
                suffix = name.split(".")[-1]
                if suffix != "lm_head" and "vision" not in name.lower():
                    suffixes.add(suffix)
        return suffixes

    def _resolve_target_modules(self, target_modules: str) -> list[str]:
        available = self._linear_suffixes()
        if not available:
            raise RuntimeError("No linear modules found for LoRA attachment.")
        if target_modules == "all_linear":
            return sorted(available)
        requested = [item.strip() for item in target_modules.split(",") if item.strip()]
        matched = [item for item in requested if item in available]
        if matched:
            return matched
        return sorted(available)

    def _attach_lora(self, model: Any, target_modules: list[str]) -> Any:
        lora_config = self._LoraConfig(
            task_type=self._TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return self._get_peft_model(model=model, peft_config=lora_config)

    def _trainable_parameters(self) -> list[tuple[str, Any]]:
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def _snapshot_trainable_state(self) -> dict[str, Any]:
        return {name: param.detach().clone().cpu() for name, param in self._trainable_parameters()}

    def _restore_trainable_state(self) -> None:
        with self._torch.no_grad():
            for name, param in self._trainable_parameters():
                initial = self._initial_lora_state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(initial)

    def _build_optimizer(self) -> Any:
        params = [param for _, param in self._trainable_parameters()]
        return self._torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def reset(self) -> None:
        self._restore_trainable_state()
        self.optimizer = self._build_optimizer()
        self._write_count = 0
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _write_payload(self, step: dict[str, Any]) -> str:
        if self.config.write_format == "raw_memory_step":
            return str(step.get("content", ""))
        if self.config.write_format == "instruction_wrapped":
            return memory_step_prompt(step)
        raise ValueError(f"Unsupported Temp-LoRA write format: {self.config.write_format}")

    def _tokenize_train_payload(self, payload: str) -> dict[str, Any]:
        raw = self.tokenizer(
            payload,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        raw_token_count = int(raw["input_ids"].shape[1])
        max_train_tokens = int(self.config.max_train_tokens or 0)
        if max_train_tokens > 0 and raw_token_count > max_train_tokens:
            input_ids = raw["input_ids"][:, :max_train_tokens].contiguous()
            attention_mask = raw.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_train_tokens].contiguous()
        else:
            input_ids = raw["input_ids"]
            attention_mask = raw.get("attention_mask")
        encoded: dict[str, Any] = {
            "input_ids": input_ids.to(self.input_device),
            "raw_token_count": raw_token_count,
            "truncated_tokens": max(0, raw_token_count - int(input_ids.shape[1])),
        }
        if attention_mask is not None:
            encoded["attention_mask"] = attention_mask.to(self.input_device)
        return encoded

    def _chunk_encoded_payload(self, encoded: dict[str, Any]) -> list[dict[str, Any]]:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        total_tokens = int(input_ids.shape[1])
        chunk_tokens = int(self.config.write_chunk_tokens or 0)
        if chunk_tokens <= 0 or total_tokens <= chunk_tokens:
            labels = input_ids.clone()
            return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]
        chunks: list[dict[str, Any]] = []
        for start in range(0, total_tokens, chunk_tokens):
            end = min(start + chunk_tokens, total_tokens)
            chunk_ids = input_ids[:, start:end].contiguous()
            chunk_mask = attention_mask[:, start:end].contiguous() if attention_mask is not None else None
            chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask, "labels": chunk_ids.clone()})
        return chunks

    def _train_batch(self, batch: dict[str, Any]) -> list[float]:
        losses: list[float] = []
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def write(self, step: dict[str, Any]) -> dict[str, Any]:
        payload = self._write_payload(step)
        encoded = self._tokenize_train_payload(payload)
        token_count = int(encoded["input_ids"].shape[1])
        raw_token_count = int(encoded.get("raw_token_count", token_count))
        truncated_tokens = int(encoded.get("truncated_tokens", 0))
        if token_count < self.config.min_train_tokens:
            return {
                "step_id": step.get("step_id"),
                "skipped": True,
                "reason": "too_few_tokens",
                "write_payload_chars": len(payload),
                "raw_write_tokens": raw_token_count,
                "write_tokens": token_count,
                "max_train_tokens": self.config.max_train_tokens,
                "truncated_by_max_train_tokens": truncated_tokens > 0,
                "truncated_tokens": truncated_tokens,
            }

        self.model.train()
        started = time.time()
        chunk_records: list[dict[str, Any]] = []
        all_losses: list[float] = []
        for chunk_index, batch in enumerate(self._chunk_encoded_payload(encoded), start=1):
            chunk_token_count = int(batch["input_ids"].shape[1])
            if chunk_token_count < self.config.min_train_tokens:
                chunk_records.append({
                    "chunk_index": chunk_index,
                    "skipped": True,
                    "reason": "too_few_tokens",
                    "write_tokens": chunk_token_count,
                })
                continue
            losses = self._train_batch(batch)
            all_losses.extend(losses)
            self._write_count += 1
            chunk_records.append({
                "chunk_index": chunk_index,
                "skipped": False,
                "write_tokens": chunk_token_count,
                "train_epochs": self.config.train_epochs,
                "losses": losses,
            })
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        return {
            "step_id": step.get("step_id"),
            "turn_start": step.get("turn_start"),
            "turn_end": step.get("turn_end"),
            "skipped": False,
            "write_payload_chars": len(payload),
            "raw_write_tokens": raw_token_count,
            "write_tokens": token_count,
            "max_train_tokens": self.config.max_train_tokens,
            "truncated_by_max_train_tokens": truncated_tokens > 0,
            "truncated_tokens": truncated_tokens,
            "write_chunk_tokens": self.config.write_chunk_tokens,
            "write_chunk_count": len(chunk_records),
            "train_epochs": self.config.train_epochs,
            "losses": all_losses,
            "chunks": chunk_records,
            "elapsed_sec": round(time.time() - started, 3),
        }

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            return prompt

    def query_with_prompt(self, prompt: str) -> QueryResult:
        started = time.time()
        self.model.eval()
        text = self._chat_text(prompt)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.input_device)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[:, encoded.input_ids.shape[1] :]
        text_out = self.tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return QueryResult(
            raw_output=text_out,
            prompt_tokens=int(encoded.input_ids.shape[1]),
            latency_sec=round(time.time() - started, 3),
            debug={
                "query_payload": prompt,
                "write_count": self._write_count,
                "target_modules": self._target_modules,
                "input_device": str(self.input_device),
                "hf_device_map": getattr(self.model, "hf_device_map", None),
            },
        )

    def _query_prompt_for_style(self, question: str) -> str:
        return query_prompt_for_style(question, self.config.query_style)

    def query(self, question: str) -> QueryResult:
        return self.query_with_prompt(self._query_prompt_for_style(question))

    def runtime_summary(self) -> dict[str, Any]:
        trainable = sum(param.numel() for _, param in self._trainable_parameters())
        return {
            "method_id": self.method_id,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "target_modules": self._target_modules,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "learning_rate": self.config.learning_rate,
            "train_epochs": self.config.train_epochs,
            "max_train_tokens": self.config.max_train_tokens,
            "write_chunk_tokens": self.config.write_chunk_tokens,
            "query_style": self.config.query_style,
            "trainable_parameters": int(trainable),
            "temp_lora_device_map": self.config.device_map,
            "temp_lora_input_device": str(self.input_device),
            "temp_lora_max_memory": list(self.config.max_memory),
            "hf_device_map": getattr(self.model, "hf_device_map", None),
            "seed": self.config.seed,
        }
