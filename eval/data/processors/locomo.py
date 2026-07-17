#!/usr/bin/env python3
"""Build LoCoMo MemQA instances with evidence-session-only context.

Policy:
- keep answerable categories 1-4 by default
- exclude category 5/adversarial by default
- allow evidence to span multiple sessions
- context and memory_steps contain only sessions that hold evidence turns
- malformed/missing evidence is reviewed and excluded, not repaired
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^D(?P<session>\d+):(?P<turn>\d+)$")
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
CATEGORY_LABELS = {
    1: "Multi",
    2: "Temp",
    3: "Open",
    4: "Single",
    5: "adversarial",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_evidence(evidence: list[str]) -> list[dict[str, int | str]] | None:
    parsed: list[dict[str, int | str]] = []
    for item in evidence or []:
        cleaned = str(item).replace("(", "").replace(")", "").strip()
        if not cleaned:
            continue
        match = EVIDENCE_RE.match(cleaned)
        if not match:
            return None
        parsed.append(
            {
                "session": int(match.group("session")),
                "turn": int(match.group("turn")),
                "dia_id": cleaned,
            }
        )
    return parsed


def context_for_session(conversation: dict[str, Any], session_num: int) -> list[dict[str, Any]]:
    session_key = f"session_{session_num}"
    date_time = conversation.get(f"{session_key}_date_time")
    out: list[dict[str, Any]] = []
    for turn in conversation.get(session_key, []):
        out.append(
            {
                "session_id": session_key,
                "session_num": session_num,
                "date_time": date_time,
                "dia_id": turn.get("dia_id"),
                "speaker": turn.get("speaker"),
                "text": turn.get("text", ""),
                "blip_caption": turn.get("blip_caption"),
                "img_url": turn.get("img_url"),
                "query": turn.get("query"),
            }
        )
    return out


def session_record(conversation: dict[str, Any], session_num: int) -> dict[str, Any]:
    session_key = f"session_{session_num}"
    return {
        "session_id": session_key,
        "session_num": session_num,
        "date_time": conversation.get(f"{session_key}_date_time"),
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
    }


def format_turn(turn: dict[str, Any]) -> str:
    text = f"{turn.get('dia_id')} {turn.get('speaker')} said: \"{turn.get('text', '')}\""
    if turn.get("blip_caption"):
        text += f" Shared image caption: {turn['blip_caption']}"
    return text


def build_memory_steps(
    contexts_by_session: dict[int, list[dict[str, Any]]],
    conversation: dict[str, Any],
    turns_per_step: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for session_num in sorted(contexts_by_session):
        context = contexts_by_session[session_num]
        session_key = f"session_{session_num}"
        date_time = conversation.get(f"{session_key}_date_time")
        for start in range(0, len(context), turns_per_step):
            chunk = context[start : start + turns_per_step]
            lines = [f"SESSION: {session_key}"]
            if date_time:
                lines.append(f"DATE: {date_time}")
            lines.extend(format_turn(turn) for turn in chunk)
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "session_id": session_key,
                    "session_num": session_num,
                    "date_time": date_time,
                    "turn_start": chunk[0].get("dia_id") if chunk else None,
                    "turn_end": chunk[-1].get("dia_id") if chunk else None,
                    "content": "\n".join(lines),
                }
            )
    return steps


def instance_id(sample_id: str, session_nums: list[int], qa_index: int) -> str:
    safe_sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))
    session_part = "-".join(str(num) for num in session_nums)
    return f"locomo_{safe_sample}__evidence_sessions_{session_part}__qa_{qa_index:04d}"


def _append_example(examples_by_skip: dict[str, list[str]], reason: str, text: str) -> None:
    if len(examples_by_skip[reason]) < 8:
        examples_by_skip[reason].append(text)


def convert(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = read_json(args.input)
    records: list[dict[str, Any]] = []
    counts = Counter()
    category_counts = Counter()
    kept_category_counts = Counter()
    skipped_reasons = Counter()
    examples_by_skip: dict[str, list[str]] = defaultdict(list)
    evidence_session_count_distribution = Counter()
    memory_step_distribution = Counter()
    context_turn_distribution = Counter()

    input_sha1 = sha1(args.input)
    input_sha256 = sha256(args.input)

    for sample in samples:
        sample_id = str(sample.get("sample_id"))
        conv = sample.get("conversation", {})
        for qa_index, qa in enumerate(sample.get("qa", [])):
            counts["qa_total"] += 1
            raw_category = qa.get("category")
            try:
                raw_category_int = int(raw_category)
            except (TypeError, ValueError):
                raw_category_int = -1
            category_counts[str(raw_category)] += 1

            evidence = qa.get("evidence") or []
            parsed = parse_evidence(evidence)
            reason = None
            if raw_category_int == 5 and not args.include_adversarial:
                reason = "adversarial_excluded"
            elif raw_category_int not in {1, 2, 3, 4, 5}:
                reason = "unknown_category"
            elif not evidence:
                reason = "missing_evidence"
            elif parsed is None:
                reason = "unparseable_evidence"
            elif not parsed:
                reason = "empty_parsed_evidence"
            if reason:
                skipped_reasons[reason] += 1
                _append_example(examples_by_skip, reason, f"{sample_id} qa_index={qa_index} category={raw_category} evidence={evidence}")
                continue

            assert parsed is not None
            session_nums = sorted({int(item["session"]) for item in parsed})
            missing_sessions = [num for num in session_nums if f"session_{num}" not in conv]
            if missing_sessions:
                skipped_reasons["missing_session"] += 1
                _append_example(examples_by_skip, "missing_session", f"{sample_id} qa_index={qa_index} sessions={missing_sessions}")
                continue

            contexts_by_session = {num: context_for_session(conv, num) for num in session_nums}
            context_ids = {turn.get("dia_id") for turns in contexts_by_session.values() for turn in turns}
            missing_evidence = [str(item["dia_id"]) for item in parsed if item["dia_id"] not in context_ids]
            if missing_evidence:
                skipped_reasons["evidence_not_in_context"] += 1
                _append_example(
                    examples_by_skip,
                    "evidence_not_in_context",
                    f"{sample_id} qa_index={qa_index} missing={missing_evidence} evidence={evidence}",
                )
                continue

            context = [turn for num in session_nums for turn in contexts_by_session[num]]
            memory_steps = build_memory_steps(contexts_by_session, conv, args.turns_per_step)
            sessions = [session_record(conv, num) for num in session_nums]
            record = {
                "task_type": "memqa",
                "dataset": "locomo",
                "split": "evidence_sessions",
                "instance_id": instance_id(sample_id, session_nums, qa_index),
                "source_sample_id": sample_id,
                "session": {
                    "session_id": "evidence_sessions",
                    "date_time": None,
                    "speaker_a": conv.get("speaker_a"),
                    "speaker_b": conv.get("speaker_b"),
                    "context_scope": "evidence_sessions_only",
                },
                "sessions": sessions,
                "context": context,
                "memory_steps": memory_steps,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": [str(item["dia_id"]) for item in parsed],
                "metadata": {
                    "raw_category": raw_category_int,
                    "category_name": CATEGORY_NAMES.get(raw_category_int),
                    "category_label": CATEGORY_LABELS.get(raw_category_int),
                    "category_name_source": "delta-Mem locomo_protocol.py mapping",
                    "is_adversarial": raw_category_int == 5,
                    "is_temporal_hint": raw_category_int == 2,
                    "source_file": args.input.name,
                    "source_sha1": input_sha1,
                    "source_sha256": input_sha256,
                    "qa_index": qa_index,
                    "evidence_sessions": session_nums,
                    "evidence_session_ids": [f"session_{num}" for num in session_nums],
                    "evidence_session_count": len(session_nums),
                    "evidence_turns": [int(item["turn"]) for item in parsed],
                    "context_scope": "evidence_sessions_only",
                    "context_turn_count": len(context),
                    "memory_step_count": len(memory_steps),
                    "filter_policy": {
                        "require_evidence": True,
                        "allow_cross_session_evidence": True,
                        "include_adversarial": args.include_adversarial,
                        "exclude_malformed_evidence_without_repair": True,
                        "context_scope": "evidence_sessions_only",
                    },
                },
            }
            records.append(record)
            counts["kept"] += 1
            kept_category_counts[str(raw_category_int)] += 1
            evidence_session_count_distribution[str(len(session_nums))] += 1
            memory_step_distribution[str(len(memory_steps))] += 1
            context_turn_distribution[str(len(context))] += 1
            if args.limit and len(records) >= args.limit:
                break
        if args.limit and len(records) >= args.limit:
            break

    meta = {
        "input": str(args.input),
        "input_sha1": input_sha1,
        "input_sha256": input_sha256,
        "output": str(args.output),
        "records": len(records),
        "counts": dict(counts),
        "raw_category_counts": dict(category_counts),
        "kept_category_counts": dict(kept_category_counts),
        "kept_category_labels": {CATEGORY_LABELS.get(int(k), k): v for k, v in kept_category_counts.items()},
        "skipped_reasons": dict(skipped_reasons),
        "skip_examples": dict(examples_by_skip),
        "evidence_session_count_distribution": dict(evidence_session_count_distribution),
        "memory_step_count_distribution": dict(memory_step_distribution),
        "context_turn_count_distribution": dict(context_turn_distribution),
        "turns_per_step": args.turns_per_step,
        "include_adversarial": args.include_adversarial,
        "limit": args.limit,
        "context_scope": "evidence_sessions_only",
        "category_mapping": CATEGORY_NAMES,
    }
    return records, meta
