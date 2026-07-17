"""Reusable memory-state-only delta-Mem wrapper for paper evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.methods.shared.memqa_io import memory_step_prompt, query_prompt_for_style
from eval.methods.shared.memory_contract import QueryResult


@dataclass(frozen=True)
class DeltaMemRuntimeConfig:
    model_path: str
    adapter_dir: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_new_tokens: int = 96
    write_format: str = "raw_memory_step"
    query_style: str = "default"


class DeltaMemRuntimeUnavailable(RuntimeError):
    pass


class DeltaMemBaseline:
    method_id = "delta_mem"

    def __init__(self, config: DeltaMemRuntimeConfig):
        self.config = config
        runtime = self._load_runtime()
        self._torch = runtime["torch"]
        self._DeltaMemChatSession = runtime["DeltaMemChatSession"]
        self._load_delta_mem_chat_model = runtime["load_delta_mem_chat_model"]
        self._get_delta_mem_online_state = runtime["get_delta_mem_online_state"]
        self._load_delta_mem_online_state = runtime["load_delta_mem_online_state"]
        self._reset_delta_mem_states = runtime["reset_delta_mem_states"]
        self.model, self.tokenizer = self._load_delta_mem_chat_model(
            model_path=config.model_path,
            device=config.device,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
            adapter_dir=config.adapter_dir,
        )
        self.session = self._DeltaMemChatSession(model=self.model, tokenizer=self.tokenizer, device=config.device)
        self.reset()

    @staticmethod
    def _load_runtime() -> dict[str, Any]:
        try:
            import torch
            from deltamem.core.delta import (
                get_delta_mem_online_state,
                load_delta_mem_online_state,
                reset_delta_mem_states,
            )
            from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model
        except Exception as exc:  # pragma: no cover - depends on external official repo.
            raise DeltaMemRuntimeUnavailable(
                "delta-mem official runtime is unavailable. Install or expose "
                "https://github.com/declare-lab/delta-Mem and use the released "
                "adapter with HFDeltaMemConfig/attach_delta_mem/load_delta_mem_adapter. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        return {
            "torch": torch,
            "DeltaMemChatSession": DeltaMemChatSession,
            "load_delta_mem_chat_model": load_delta_mem_chat_model,
            "get_delta_mem_online_state": get_delta_mem_online_state,
            "load_delta_mem_online_state": load_delta_mem_online_state,
            "reset_delta_mem_states": reset_delta_mem_states,
        }

    @classmethod
    def runtime_check(cls) -> dict[str, Any]:
        try:
            cls._load_runtime()
            return {"available": True, "error": None}
        except DeltaMemRuntimeUnavailable as exc:
            return {"available": False, "error": str(exc)}

    def reset(self) -> None:
        self.session.reset()

    def _clear_context_preserve_delta_state(self) -> dict[str, Any]:
        state = self._get_delta_mem_online_state(self.model)
        stats_before = self.session.state_stats()
        self.session.messages = []
        self.session.processed_input_ids = None
        self.session.past_key_values = None
        self.session.last_ingest_stats = {}
        self.session.last_decode_stats = {}
        self.session.last_turn_stats = {}
        self.session.cached_message_input_ids = None
        self.session.cached_write_message_ids = None
        self.session.cached_write_sentence_ids = None
        self.session.cached_message_count = 0
        self._reset_delta_mem_states(self.model)
        self._load_delta_mem_online_state(self.model, state)
        return {
            "state_stats_before_context_clear": stats_before,
            "state_stats_after_context_clear": self.session.state_stats(),
            "kept_messages": False,
        }

    def _write_payload(self, step: dict[str, Any]) -> str:
        if self.config.write_format == "raw_memory_step":
            return str(step.get("content", ""))
        if self.config.write_format == "instruction_wrapped":
            return memory_step_prompt(step)
        raise ValueError(f"Unsupported delta-mem write format: {self.config.write_format}")

    def write(self, step: dict[str, Any]) -> dict[str, Any]:
        payload = self._write_payload(step)
        self.session.messages = [{"role": "user", "content": payload}]
        self.session.processed_input_ids = None
        self.session.past_key_values = None
        full_ids = self.session._tokenize_messages(self.session.messages, add_generation_prompt=False)
        self.session._ingest_full_ids(full_ids)
        ingest_stats = dict(self.session.last_ingest_stats)
        clear_report = self._clear_context_preserve_delta_state()
        return {
            "step_id": step.get("step_id"),
            "turn_start": step.get("turn_start"),
            "turn_end": step.get("turn_end"),
            "write_payload_chars": len(payload),
            "write_tokens": int(full_ids.shape[1]),
            "ingest_stats": ingest_stats,
            "context_clear_report": clear_report,
        }

    def query(self, question: str) -> QueryResult:
        # Make query isolation idempotent even if the caller wrote all steps first.
        self._clear_context_preserve_delta_state()
        payload = query_prompt_for_style(question, self.config.query_style)
        result = self.session.generate_reply(
            payload,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            write_enabled=False,
            include_debug=True,
        )
        turn_stats = result.get("turn_stats", {}) if isinstance(result, dict) else {}
        prompt_tokens = int(turn_stats.get("prompt_ingest", {}).get("full_tokens", 0))
        latency = float(turn_stats.get("elapsed_ms", 0.0)) / 1000.0 if turn_stats else 0.0
        return QueryResult(
            raw_output=str(result.get("assistant_display", result.get("assistant", ""))).strip(),
            prompt_tokens=prompt_tokens,
            latency_sec=round(latency, 3),
            debug={"query_payload": payload, "turn_stats": turn_stats, "state_stats": self.session.state_stats()},
        )
