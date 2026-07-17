"""MemQA consumer helpers for shared memory baselines."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        return records[:limit]
    return records


def safe_label(text: str) -> str:
    return text.replace("/", "_").replace(":", "_").replace(" ", "_")


def memory_step_prompt(step: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Conversation memory segment.",
            "Write the following dated dialogue segment into the online memory state for later question answering.",
            "Do not answer now.",
            "",
            str(step.get("content", "")),
        ]
    )


def query_prompt(question: str) -> str:
    return "\n".join(
        [
            f"Question: {question}",
            'Answer with a short phrase using only the online memory state. If the answer is not known from memory, say "No information available".',
        ]
    )


def memory_direct_query_prompt(question: str) -> str:
    return "\n".join(
        [
            "Answer from the learned memory state produced during the information phase.",
            "Give the shortest factual answer you can. Do not explain.",
            f"Question: {question}",
            "Short answer:",
        ]
    )


def minimal_query_prompt(question: str) -> str:
    return "\n".join([f"Question: {question}", "Answer only the short answer."])


def query_prompt_for_style(question: str, style: str) -> str:
    if style == "default":
        return query_prompt(question)
    if style == "memory_direct":
        return memory_direct_query_prompt(question)
    if style == "minimal":
        return minimal_query_prompt(question)
    raise ValueError(f"Unsupported query_style: {style}")


def stable_text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _forbidden_fragments(instance: dict[str, Any]) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    allowed_question = str(instance.get("question", ""))
    for step in instance.get("memory_steps", []):
        content = str(step.get("content", "")).strip()
        if len(content) >= 40:
            fragments.append(("memory_step", content[:120]))
    for turn in instance.get("context", []):
        text = str(turn.get("text", "")).strip()
        if len(text) >= 40:
            fragments.append(("context_turn", text[:120]))
        caption = str(turn.get("blip_caption") or "").strip()
        if len(caption) >= 40:
            fragments.append(("context_caption", caption[:120]))
    answer = str(instance.get("answer", "")).strip()
    if len(answer) >= 20 and answer not in allowed_question:
        fragments.append(("gold_answer", answer[:120]))
    for evidence in instance.get("evidence", []) or []:
        evidence_text = str(evidence).strip()
        if evidence_text:
            fragments.append(("evidence_id", evidence_text))
    return fragments


def audit_query_payload(instance: dict[str, Any], query_text: str) -> list[str]:
    """Return policy issues if query_text leaks write/context/gold material."""

    issues: list[str] = []
    for label, fragment in _forbidden_fragments(instance):
        if fragment and fragment in query_text:
            issues.append(f"query payload includes forbidden {label}: {fragment[:80]!r}")
    return issues


def output_record(
    *,
    run_id: str,
    baseline: str,
    model_label: str,
    model_path: str,
    instance: dict[str, Any],
    instance_index: int,
    instance_count: int,
    raw_output: str,
    prompt_tokens: int,
    latency_sec: float,
    context_policy: str,
    generation_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": run_id,
        "date": utc_now(),
        "task": "memqa",
        "dataset": instance.get("dataset"),
        "split": instance.get("split"),
        "baseline": baseline,
        "model_label": model_label,
        "model_path": model_path,
        "instance_index": instance_index,
        "instance_count": instance_count,
        "instance_id": instance["instance_id"],
        "source_sample_id": instance.get("source_sample_id"),
        "source_dataset": instance.get("metadata", {}).get("source_dataset"),
        "raw_category": instance.get("metadata", {}).get("raw_category"),
        "is_adversarial": instance.get("metadata", {}).get("is_adversarial"),
        "question": instance["question"],
        "context_policy": context_policy,
        "prompt_tokens": prompt_tokens,
        "latency_sec": latency_sec,
        "raw_output": raw_output,
        "generation_config": generation_config,
    }
    if extra:
        record.update(extra)
    return record
