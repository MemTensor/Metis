#!/usr/bin/env python3
"""Normalize MetisOps v23 artifacts into MemOP JSONL settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


EXCLUDED_OPERATIONS = {"trajectory_ops"}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_real_zip_member(name: str) -> bool:
    basename = os.path.basename(name)
    return not name.startswith("__MACOSX/") and not basename.startswith("._") and basename != ".DS_Store" and not name.endswith("/")


def normalize_operation(value: str | None) -> str:
    mapping = {
        "remember": "remember",
        "update": "update",
        "forget": "forget",
        "reflect": "reflect",
        "trajectoryops": "trajectory_ops",
    }
    raw = (value or "").replace("_", "").lower()
    return mapping.get(raw, (value or "unknown").lower())


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "")).upper() or "MESSAGE"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_segment(segment: dict[str, Any]) -> str:
    header = f"SEGMENT {segment.get('segment_index')}: {segment.get('segment_role', '')}".strip()
    return header + "\n" + format_messages(segment.get("dialogue", []))


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def find_message_index(segment: dict[str, Any], provenance: dict[str, Any]) -> tuple[int | None, str]:
    dialogue = segment.get("dialogue", [])
    quote = str(provenance.get("quote", "")).strip()
    quote_norm = normalize_text(quote)
    if quote_norm:
        for idx, message in enumerate(dialogue):
            content_norm = normalize_text(str(message.get("content", "")))
            if quote_norm in content_norm or content_norm in quote_norm:
                return idx, "quote_match"
    turn_index = provenance.get("turn_index")
    if isinstance(turn_index, int):
        for candidate in (turn_index - 1, turn_index):
            if 0 <= candidate < len(dialogue):
                return candidate, "turn_index_fallback"
    return None, "unmatched"


def pair_for_message(segment: dict[str, Any], message_index: int) -> tuple[int, int]:
    dialogue = segment.get("dialogue", [])
    role = dialogue[message_index].get("role")
    if role == "user" and message_index + 1 < len(dialogue):
        return message_index, message_index + 1
    if role == "assistant" and message_index - 1 >= 0:
        return message_index - 1, message_index
    start = message_index if message_index % 2 == 0 else max(0, message_index - 1)
    end = min(len(dialogue) - 1, start + 1)
    return start, end


def build_full_segment_record(
    source_file: str,
    source_stem: str,
    obj: dict[str, Any],
    answer: dict[str, Any],
    answer_index: int,
) -> dict[str, Any]:
    operation = normalize_operation(obj.get("operation_type"))
    context = []
    memory_steps = []
    for step_id, segment in enumerate(obj.get("conversations", []), start=1):
        segment_id = f"S{segment.get('segment_index')}"
        content = format_segment(segment)
        context.append(
            {
                "segment_id": segment_id,
                "segment_index": segment.get("segment_index"),
                "segment_role": segment.get("segment_role"),
                "messages": segment.get("dialogue", []),
                "content": content,
            }
        )
        memory_steps.append(
            {
                "step_id": step_id,
                "turn_start": segment_id,
                "turn_end": segment_id,
                "commit_granularity": "eight_utterance_segment",
                "content": content,
            }
        )

    return base_record(
        source_file=source_file,
        source_stem=source_stem,
        obj=obj,
        answer=answer,
        answer_index=answer_index,
        setting="full_segments",
        context=context,
        memory_steps=memory_steps,
        evidence=[f"S{span.get('segment_index')}:T{span.get('turn_index')}" for span in answer.get("gold_provenance", [])],
        operation=operation,
        extraction_audit={"unmatched_provenance": 0, "match_modes": []},
    )


def build_gold_turn_record(
    source_file: str,
    source_stem: str,
    obj: dict[str, Any],
    answer: dict[str, Any],
    answer_index: int,
) -> dict[str, Any]:
    operation = normalize_operation(obj.get("operation_type"))
    segments = {segment.get("segment_index"): segment for segment in obj.get("conversations", [])}
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    unmatched = 0
    match_modes: list[str] = []

    for provenance in answer.get("gold_provenance", []) or []:
        segment_index = provenance.get("segment_index")
        segment = segments.get(segment_index)
        if not segment:
            unmatched += 1
            continue
        message_index, mode = find_message_index(segment, provenance)
        match_modes.append(mode)
        if message_index is None:
            unmatched += 1
            continue
        start, end = pair_for_message(segment, message_index)
        selected[(int(segment_index), start)] = {
            "segment": segment,
            "start": start,
            "end": end,
            "provenance": provenance,
            "match_mode": mode,
        }

    context = []
    memory_steps = []
    for step_id, ((segment_index, start), item) in enumerate(sorted(selected.items()), start=1):
        segment = item["segment"]
        messages = segment.get("dialogue", [])[item["start"] : item["end"] + 1]
        turn_id = f"S{segment_index}:M{item['start'] + 1}-{item['end'] + 1}"
        content = f"SEGMENT {segment_index} EVIDENCE TURN\n" + format_messages(messages)
        context.append(
            {
                "turn_id": turn_id,
                "segment_index": segment_index,
                "segment_role": segment.get("segment_role"),
                "message_start_index": item["start"] + 1,
                "message_end_index": item["end"] + 1,
                "messages": messages,
                "content": content,
                "match_mode": item["match_mode"],
                "provenance": item["provenance"],
            }
        )
        memory_steps.append(
            {
                "step_id": step_id,
                "turn_start": turn_id,
                "turn_end": turn_id,
                "commit_granularity": "two_utterance_gold_turn",
                "content": content,
            }
        )

    return base_record(
        source_file=source_file,
        source_stem=source_stem,
        obj=obj,
        answer=answer,
        answer_index=answer_index,
        setting="gold_turns",
        context=context,
        memory_steps=memory_steps,
        evidence=[item["turn_id"] for item in context],
        operation=operation,
        extraction_audit={"unmatched_provenance": unmatched, "match_modes": match_modes},
    )


def base_record(
    source_file: str,
    source_stem: str,
    obj: dict[str, Any],
    answer: dict[str, Any],
    answer_index: int,
    setting: str,
    context: list[dict[str, Any]],
    memory_steps: list[dict[str, Any]],
    evidence: list[str],
    operation: str,
    extraction_audit: dict[str, Any],
) -> dict[str, Any]:
    pair_id = answer.get("question_pair_id") or f"q{answer_index:02d}"
    instance_id = f"metisops_v23_{setting}_{source_stem}__{slug(str(pair_id))}"
    return {
        "task_type": "memop",
        "dataset": "metisops_v23",
        "split": "test",
        "setting": setting,
        "instance_id": instance_id,
        "source_sample_id": source_stem,
        "operation": operation,
        "subtask": answer.get("evaluation_category") or answer.get("evaluation_type"),
        "context": context,
        "memory_steps": memory_steps,
        "question": answer.get("question", ""),
        "answer": answer.get("expected_answer", ""),
        "rubric": {
            "judge_type": "semantic_consistency_with_rubric",
            "judge_rubric": answer.get("judge_rubric", {}),
            "diagnostic_checks": answer.get("diagnostic_checks", {}),
        },
        "evidence": evidence,
        "metadata": {
            "branch_id": "metisops_v23",
            "source_file": source_file,
            "operation_type_raw": obj.get("operation_type"),
            "target_fact": obj.get("target_fact"),
            "question_pair_id": pair_id,
            "answer_index": answer_index,
            "source_evaluation_setting": answer.get("evaluation_setting"),
            "evaluation_type": answer.get("evaluation_type"),
            "evaluation_category": answer.get("evaluation_category"),
            "difficulty": answer.get("difficulty"),
            "state_transition_probe_type": answer.get("state_transition_probe_type", ""),
            "application_probe_type": answer.get("application_probe_type", ""),
            "gold_memory_state": answer.get("gold_memory_state", ""),
            "gold_provenance": answer.get("gold_provenance", []),
            "difficulty_knobs": obj.get("difficulty_knobs", {}),
            "extraction_audit": extraction_audit,
            "context_policy": "No injected distractor artifact is used. Query sees question only for memory_only baselines.",
        },
    }

def build_records(zip_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_records: list[dict[str, Any]] = []
    gold_turn_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if is_real_zip_member(name)]
        conversation_names = sorted(
            name
            for name in names
            if "/2-evidence_conversation_v23_30topic/" in name
            and name.endswith(".json")
            and "llm_verify" not in name
            and "_attempt_" not in name
        )
        injected_names = sorted(
            name
            for name in names
            if "/4-inject_evidence_with_distractors_v23_30topic/" in name
            and name.endswith(".json")
            and "llm_verify" not in name
            and "_attempt_" not in name
        )
        for name in conversation_names:
            obj = json.loads(archive.read(name).decode("utf-8"))
            source_file = os.path.basename(name)
            source_stem = Path(source_file).stem
            operation = normalize_operation(obj.get("operation_type"))
            if operation in EXCLUDED_OPERATIONS:
                skipped.append({"source_file": source_file, "operation": operation, "reason": "excluded_operation"})
                continue
            conversations = obj.get("conversations", [])
            if len(conversations) != 3 or any(len(segment.get("dialogue", [])) != 8 for segment in conversations):
                skipped.append({"source_file": source_file, "reason": "unexpected_conversation_shape"})
                continue
            answers = [answer for answer in obj.get("answer", []) if answer.get("evaluation_setting") == "longitudinal_operation"]
            for answer_index, answer in enumerate(answers, start=1):
                full_records.append(build_full_segment_record(source_file, source_stem, obj, answer, answer_index))
                gold_turn_records.append(build_gold_turn_record(source_file, source_stem, obj, answer, answer_index))

        meta = {
        "zip_sha256": sha256_file(zip_path),
        "conversation_file_count": len(conversation_names),
        "injected_distractor_file_count": len(injected_names),
        "excluded_operations": sorted(EXCLUDED_OPERATIONS),
        "skipped": skipped,
    }
    return full_records, gold_turn_records, meta


def length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    step_counts = [len(record["memory_steps"]) for record in records]
    utterance_counts = [sum(len(item.get("messages", [])) for item in record["context"]) for record in records]
    chars = [sum(len(item.get("content", "")) for item in record["context"]) for record in records]
    return {
        "records": len(records),
        "memory_steps_min": min(step_counts) if step_counts else 0,
        "memory_steps_max": max(step_counts) if step_counts else 0,
        "memory_steps_avg": round(sum(step_counts) / len(step_counts), 3) if step_counts else 0,
        "utterances_avg": round(sum(utterance_counts) / len(utterance_counts), 3) if utterance_counts else 0,
        "utterances_min": min(utterance_counts) if utterance_counts else 0,
        "utterances_max": max(utterance_counts) if utterance_counts else 0,
        "context_chars_avg": round(sum(chars) / len(chars), 1) if chars else 0,
        "context_chars_max": max(chars) if chars else 0,
    }
