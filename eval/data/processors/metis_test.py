#!/usr/bin/env python3
"""Normalize the MetisTest v2.4 MemoryOps test subset into MemOP JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def as_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def normalize_operation(value: str | None) -> str:
    mapping = {"reflection": "reflect"}
    return mapping.get((value or "").lower(), (value or "unknown").lower())


def format_turn(turn: list[dict[str, Any]]) -> str:
    lines = []
    for message in turn:
        role = str(message.get("role", "")).upper() or "MESSAGE"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def split_turn_fragments(turn: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    fragments: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(turn):
        current = turn[index]
        nxt = turn[index + 1] if index + 1 < len(turn) else None
        if current.get("role") == "user" and isinstance(nxt, dict) and nxt.get("role") == "assistant":
            fragments.append([current, nxt])
            index += 2
        else:
            fragments.append([current])
            index += 1
    return fragments


def fragment_granularity(fragment: list[dict[str, Any]]) -> str:
    roles = [message.get("role") for message in fragment]
    if roles == ["user", "assistant"]:
        return "one_user_assistant_pair"
    return "source_message_fragment"


def role_text(turn: list[dict[str, Any]], role: str) -> str:
    parts = [str(message.get("content", "")).strip() for message in turn if message.get("role") == role]
    return "\n".join(part for part in parts if part)


def build_records(source_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_path = source_dir / "all.jsonl"
    index_path = source_dir / "index.jsonl"
    manifest_path = source_dir / "manifest.json"
    raw_records = read_jsonl(all_path)
    index_records = read_jsonl(index_path)
    index_by_sample = {record["sample_id"]: record for record in index_records}
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    normalized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for raw_index, raw in enumerate(raw_records):
        sample_id = raw["sample_id"]
        metadata = raw.get("metadata", {})
        index_meta = index_by_sample.get(sample_id, {})
        messages = raw.get("messages", [])
        query_ids = as_list(raw.get("query_turn_id"))
        query_id_set = set(query_ids)
        reference_ids = as_list(raw.get("reference_turn_id"))
        operation = normalize_operation(index_meta.get("operation") or metadata.get("type"))

        for query_id in query_ids:
            if query_id < 0 or query_id >= len(messages):
                skipped.append({"sample_id": sample_id, "query_turn_id": query_id, "reason": "query_turn_id_out_of_range"})
                continue
            query_turn = messages[query_id]
            if not isinstance(query_turn, list):
                skipped.append({"sample_id": sample_id, "query_turn_id": query_id, "reason": "query_turn_is_not_message_pair"})
                continue

            memory_turn_ids = [idx for idx in range(query_id) if idx not in query_id_set]
            context = []
            memory_steps = []
            step_id = 1
            fragment_source_ids: list[str] = []
            non_pair_fragments = 0
            for turn_id in memory_turn_ids:
                turn = messages[turn_id]
                content = format_turn(turn)
                context.append(
                    {
                        "turn_id": f"T{turn_id}",
                        "source_turn_index": turn_id,
                        "messages": turn,
                        "content": content,
                        "is_reference_turn": turn_id in reference_ids,
                    }
                )
                for fragment_index, fragment in enumerate(split_turn_fragments(turn), start=1):
                    fragment_id = f"T{turn_id}F{fragment_index}"
                    granularity = fragment_granularity(fragment)
                    if granularity != "one_user_assistant_pair":
                        non_pair_fragments += 1
                    fragment_source_ids.append(fragment_id)
                    memory_steps.append(
                        {
                            "step_id": step_id,
                            "turn_start": fragment_id,
                            "turn_end": fragment_id,
                            "source_turn_id": f"T{turn_id}",
                            "commit_granularity": granularity,
                            "messages": fragment,
                            "content": format_turn(fragment),
                        }
                    )
                    step_id += 1

            question = role_text(query_turn, "user")
            answer = role_text(query_turn, "assistant")
            if not question or not answer:
                skipped.append({"sample_id": sample_id, "query_turn_id": query_id, "reason": "missing_question_or_answer"})
                continue

            suffix = f"q{query_id:02d}"
            normalized.append(
                {
                    "task_type": "memop",
                    "dataset": "metis_test",
                    "split": "test",
                    "setting": "turn_commit",
                    "instance_id": f"metistest_{sample_id}__{suffix}",
                    "source_sample_id": sample_id,
                    "operation": operation,
                    "subtask": index_meta.get("bucket") or "unknown",
                    "context": context,
                    "memory_steps": memory_steps,
                    "question": question,
                    "answer": answer,
                    "rubric": {
                        "judge_type": "semantic_consistency",
                        "source": "assistant_message_as_gold",
                    },
                    "evidence": [f"T{idx}" for idx in reference_ids if idx in memory_turn_ids],
                    "metadata": {
                        "branch_id": "metis_test",
                        "source_dir": str(source_dir),
                        "source_file": index_meta.get("source_file") or metadata.get("v2_release_source_file"),
                        "target_file": index_meta.get("target_file"),
                        "source_line": index_meta.get("source_line"),
                        "raw_index": raw_index,
                        "subset_index": index_meta.get("subset_index"),
                        "operation_raw": metadata.get("type"),
                        "style": metadata.get("style"),
                        "view": metadata.get("view"),
                        "subtype": index_meta.get("subtype"),
                        "mixed_source_task_type": index_meta.get("mixed_source_task_type"),
                        "reference_turn_id": raw.get("reference_turn_id"),
                        "query_turn_id": query_id,
                        "all_query_turn_ids": query_ids,
                        "memory_turn_ids": memory_turn_ids,
                        "memory_step_source_ids": fragment_source_ids,
                        "non_pair_memory_fragments": non_pair_fragments,
                        "commit_policy": "Commit every source user/assistant pair before the query, excluding query turns. If a source turn has an unpaired message, preserve it as a marked source_message_fragment rather than dropping source content.",
                        "raw_metadata": metadata,
                    },
                }
            )

    meta = {
        "source_manifest": source_manifest,
        "source_all_sha256": sha256_file(all_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "raw_records": len(raw_records),
        "normalized_records": len(normalized),
        "skipped": skipped,
    }
    return normalized, meta


def summarize_lengths(records: list[dict[str, Any]]) -> dict[str, Any]:
    step_counts = [len(record["memory_steps"]) for record in records]
    context_chars = [sum(len(item["content"]) for item in record["context"]) for record in records]
    return {
        "memory_step_count_min": min(step_counts) if step_counts else 0,
        "memory_step_count_max": max(step_counts) if step_counts else 0,
        "memory_step_count_avg": round(sum(step_counts) / len(step_counts), 3) if step_counts else 0,
        "context_chars_avg": round(sum(context_chars) / len(context_chars), 1) if context_chars else 0,
        "context_chars_max": max(context_chars) if context_chars else 0,
    }
