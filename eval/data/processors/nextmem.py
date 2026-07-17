#!/usr/bin/env python3
"""Normalize the official NextMem Task 2 STM source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_CONFIG_DATASETS = ["squad", "hotpot", "race", "longmemeval", "locomo"]
OFFICIAL_TASK2_RUNNER_DATASETS = ["hotpot", "squad", "locomo", "longmemeval"]
OFFICIAL_SAMPLE_GAP = {
    "longmemeval": 1,
    "locomo": 1,
    "squad": 8,
    "hotpot": 4,
}


@dataclass(frozen=True)
class SourceRecord:
    source_dataset: str
    source_file: Path
    raw_index: int
    official_sample_index: int | None
    item: dict[str, Any]
    joined_reference: str
    official_reference: str
    reference_word_count: int
    reference_token_count: int | None
    official_filter_pass: bool
    empty_reference: bool


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_name(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("stm_") or not stem.endswith("_test"):
        raise ValueError(f"Unexpected NextMem STM filename: {path.name}")
    return stem.removeprefix("stm_").removesuffix("_test")


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def canonical_answer(answer: Any) -> tuple[str, str]:
    if answer is None:
        return "", "null_to_empty_string"
    if isinstance(answer, str):
        return answer, "string_identity"
    if isinstance(answer, (int, float, bool)):
        return str(answer), "scalar_to_string"
    if isinstance(answer, list):
        if not answer:
            return "", "empty_list_to_empty_string"
        if all(isinstance(item, str) for item in answer):
            return answer[0], "string_list_first_item"
        return json.dumps(answer[0], ensure_ascii=False), "list_first_item_json"
    return json.dumps(answer, ensure_ascii=False, sort_keys=True), "object_json_string"


def stat(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def pct(q: float) -> int:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        if lo == hi:
            return ordered[lo]
        return round(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": pct(0.25),
        "median": round(statistics.median(ordered)),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
        "mean": round(statistics.mean(ordered), 2),
    }


def load_tokenizer(path: Path | None) -> Any | None:
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"Tokenizer unavailable: {exc}")
        return None
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def token_count(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def reference_fields(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    issues = []
    refs = item.get("references")
    if not isinstance(refs, list):
        issues.append(f"references_type={type_name(refs)}")
        return [], issues
    out = []
    for ref in refs:
        if not isinstance(ref, str):
            issues.append(f"reference_item_type={type_name(ref)}")
        out.append(str(ref))
    return out, issues


def selected_raw_indices(total: int, source_dataset: str) -> list[tuple[int, int]]:
    gap = OFFICIAL_SAMPLE_GAP.get(source_dataset)
    if gap is None:
        return []
    return [(official_index, raw_index) for official_index, raw_index in enumerate(range(0, total, gap))]


def collect_records(raw_dir: Path, tokenizer: Any | None) -> tuple[list[SourceRecord], dict[str, Any]]:
    all_records: list[SourceRecord] = []
    source_files = sorted(raw_dir.glob("stm_*_test.json"))
    file_manifest = {}
    field_issues: dict[str, list[str]] = defaultdict(list)
    raw_counts = {}

    for path in source_files:
        source_dataset = dataset_name(path)
        data = read_json(path)
        if not isinstance(data, list):
            raise ValueError(f"{path} should contain a list, got {type_name(data)}")
        raw_counts[source_dataset] = len(data)
        file_manifest[source_dataset] = {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "records": len(data),
            "in_official_config": source_dataset in OFFICIAL_CONFIG_DATASETS,
            "in_official_task2_runner": source_dataset in OFFICIAL_TASK2_RUNNER_DATASETS,
            "official_sample_gap": OFFICIAL_SAMPLE_GAP.get(source_dataset),
        }
        sample_index_by_raw = {raw_index: official for official, raw_index in selected_raw_indices(len(data), source_dataset)}
        for raw_index, item in enumerate(data):
            if not isinstance(item, dict):
                field_issues[source_dataset].append(f"raw_index={raw_index} item_type={type_name(item)}")
                continue
            missing = [key for key in ("question", "answer", "references") if key not in item]
            if missing:
                field_issues[source_dataset].append(f"raw_index={raw_index} missing={missing}")
            refs, ref_issues = reference_fields(item)
            for issue in ref_issues:
                field_issues[source_dataset].append(f"raw_index={raw_index} {issue}")
            joined = "\n".join(refs)
            empty_reference = joined == ""
            official_reference = "None" if empty_reference else joined
            word_count = len(official_reference.split())
            all_records.append(
                SourceRecord(
                    source_dataset=source_dataset,
                    source_file=path,
                    raw_index=raw_index,
                    official_sample_index=sample_index_by_raw.get(raw_index),
                    item=item,
                    joined_reference=joined,
                    official_reference=official_reference,
                    reference_word_count=word_count,
                    reference_token_count=token_count(tokenizer, official_reference),
                    official_filter_pass=word_count < 128,
                    empty_reference=empty_reference,
                )
            )

    meta = {
        "raw_dir": str(raw_dir),
        "raw_counts": raw_counts,
        "file_manifest": file_manifest,
        "field_issues": dict(field_issues),
        "official_config_datasets": OFFICIAL_CONFIG_DATASETS,
        "official_task2_runner_datasets": OFFICIAL_TASK2_RUNNER_DATASETS,
        "official_sample_gap": OFFICIAL_SAMPLE_GAP,
        "missing_config_files": [
            name for name in OFFICIAL_CONFIG_DATASETS if not (raw_dir / f"stm_{name}_test.json").exists()
        ],
        "extra_stm_files": [
            dataset_name(path)
            for path in source_files
            if dataset_name(path) not in OFFICIAL_CONFIG_DATASETS
        ],
    }
    return all_records, meta


def record_group(records: list[SourceRecord], *, official_only: bool = False, filtered: bool = False) -> list[SourceRecord]:
    out = records
    if official_only:
        out = [record for record in out if record.official_sample_index is not None]
    if filtered:
        out = [record for record in out if record.official_filter_pass]
    return out


def answer_in_reference(record: SourceRecord) -> bool:
    answer, _ = canonical_answer(record.item.get("answer"))
    answer = answer.strip().lower()
    if not answer:
        return False
    return answer in record.official_reference.lower()


def summarize_group(records: list[SourceRecord]) -> dict[str, Any]:
    answers = [canonical_answer(record.item.get("answer"))[0] for record in records]
    token_values = [record.reference_token_count for record in records if record.reference_token_count is not None]
    return {
        "records": len(records),
        "reference_word_count": stat([record.reference_word_count for record in records]),
        "reference_token_count": stat(token_values),
        "reference_count": stat([len(record.item.get("references", [])) for record in records]),
        "question_word_count": stat([len(str(record.item.get("question", "")).split()) for record in records]),
        "answer_word_count": stat([len(answer.split()) for answer in answers]),
        "empty_references": sum(record.empty_reference for record in records),
        "answer_type_counts": dict(Counter(type_name(record.item.get("answer")) for record in records)),
        "answer_exact_substring_in_reference": sum(answer_in_reference(record) for record in records),
    }


def build_audit(records: list[SourceRecord], source_meta: dict[str, Any]) -> dict[str, Any]:
    by_dataset: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.source_dataset].append(record)

    datasets = {}
    for source_dataset in sorted(by_dataset):
        dataset_records = by_dataset[source_dataset]
        sampled = record_group(dataset_records, official_only=True)
        sampled_filtered = record_group(dataset_records, official_only=True, filtered=True)
        datasets[source_dataset] = {
            "raw_records": len(dataset_records),
            "official_sample_gap": OFFICIAL_SAMPLE_GAP.get(source_dataset),
            "after_official_sample_gap": len(sampled),
            "after_official_filter_ref_words_lt_128": len(sampled_filtered),
            "skipped_by_official_filter": len(sampled) - len(sampled_filtered),
            "in_official_config": source_dataset in OFFICIAL_CONFIG_DATASETS,
            "in_official_task2_runner": source_dataset in OFFICIAL_TASK2_RUNNER_DATASETS,
            "all_raw_stats": summarize_group(dataset_records),
            "official_sampled_filtered_stats": summarize_group(sampled_filtered),
        }

    examples = {}
    for source_dataset in sorted(by_dataset):
        pool = record_group(by_dataset[source_dataset], official_only=True, filtered=True)
        if not pool:
            pool = by_dataset[source_dataset][:5]
        examples[source_dataset] = [
            {
                "source_sample_id": f"{record.source_dataset}:{record.raw_index}",
                "official_sample_index": record.official_sample_index,
                "question": record.item.get("question"),
                "answer": record.item.get("answer"),
                "answer_type": type_name(record.item.get("answer")),
                "canonical_answer": canonical_answer(record.item.get("answer"))[0],
                "reference_count": len(record.item.get("references", [])),
                "reference_word_count": record.reference_word_count,
                "reference_token_count": record.reference_token_count,
                "answer_exact_substring_in_reference": answer_in_reference(record),
                "references_preview": record.official_reference[:500],
            }
            for record in pool[:5]
        ]

    official_default_filtered = [
        record
        for record in records
        if record.source_dataset in OFFICIAL_TASK2_RUNNER_DATASETS
        and record.official_sample_index is not None
        and record.official_filter_pass
    ]
    return {
        "source": source_meta,
        "datasets": datasets,
        "official_default_filtered_summary": summarize_group(official_default_filtered),
        "examples": examples,
        "automated_judgment": {
            "short_enough_for_first_smoke": max(
                [record.reference_word_count for record in official_default_filtered] or [0]
            )
            < 128,
            "field_shape_matches_question_answer_references": not any(source_meta["field_issues"].values()),
            "token_stats_available": bool(
                summarize_group(official_default_filtered)["reference_token_count"].get("count")
            ),
            "answer_support_signal": (
                "exact_substring_only; manual review still required for paraphrases and yes/no answers"
            ),
        },
    }


def make_instance(record: SourceRecord, split: str) -> dict[str, Any]:
    refs = record.item.get("references") if isinstance(record.item.get("references"), list) else []
    context = [{"ref_id": f"R{index + 1}", "text": str(ref)} for index, ref in enumerate(refs)]
    if record.empty_reference:
        context = []
    answer, policy = canonical_answer(record.item.get("answer"))
    ref_start = context[0]["ref_id"] if context else None
    ref_end = context[-1]["ref_id"] if context else None
    return {
        "task_type": "memqa",
        "dataset": "nextmem_stm",
        "split": split,
        "instance_id": f"nextmem_stm_{record.source_dataset}_{record.raw_index:06d}",
        "source_sample_id": f"{record.source_dataset}:{record.raw_index}",
        "context": context,
        "memory_steps": [
            {
                "step_id": 1,
                "turn_start": ref_start,
                "turn_end": ref_end,
                "content": f"REFERENCE:\n{record.official_reference}",
            }
        ],
        "question": record.item.get("question"),
        "answer": answer,
        "evidence": [item["ref_id"] for item in context],
        "metadata": {
            "branch_id": "nextmem_stm",
            "source_dataset": record.source_dataset,
            "source_file": record.source_file.name,
            "source_sha256": sha256(record.source_file),
            "raw_index": record.raw_index,
            "official_sample_index": record.official_sample_index,
            "official_task": "task2_contextual_generation",
            "official_dataset_in_config": record.source_dataset in OFFICIAL_CONFIG_DATASETS,
            "official_dataset_in_task2_runner": record.source_dataset in OFFICIAL_TASK2_RUNNER_DATASETS,
            "official_sample_gap": OFFICIAL_SAMPLE_GAP.get(record.source_dataset),
            "official_filter_ref_words_lt_128": True,
            "official_filter_pass": record.official_filter_pass,
            "official_empty_reference_substitution": record.empty_reference,
            "reference_count": len(refs),
            "reference_word_count": record.reference_word_count,
            "reference_token_count": record.reference_token_count,
            "raw_answer": record.item.get("answer"),
            "raw_answer_type": type_name(record.item.get("answer")),
            "canonical_answer_policy": policy,
            "question_type": record.item.get("question_type"),
            "normalization_policy": "one_memory_step_joined_references_matching_official_task2_encode_call",
            "query_policy": "memory_only_query_gets_question_only; references/evidence/gold are forbidden at query time",
        },
    }


def build_split(records: list[SourceRecord], split_kind: str, per_dataset: int) -> list[dict[str, Any]]:
    out = []
    for source_dataset in OFFICIAL_TASK2_RUNNER_DATASETS:
        candidates = [
            record
            for record in records
            if record.source_dataset == source_dataset
            and record.official_sample_index is not None
            and record.official_filter_pass
        ]
        selected = candidates if split_kind == "full" else candidates[:per_dataset]
        split = "task2_contextual_generation_test" if split_kind == "full" else "task2_contextual_generation_smoke"
        out.extend(make_instance(record, split) for record in selected)
    return out
