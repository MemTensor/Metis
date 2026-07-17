#!/usr/bin/env python3
"""Normalize the ATM and MemDaily source payloads used by the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ATM_REVISION = "78e826dc07e97466b2f54443831ef9a83ab8b27c"
MEMDAILY_REVISION = "db9d5d552d6cb1d859f692eb7e6c0fd6d61d3815"
PROTOCOL_VERSION = "gold-v1-20260713"
MEMDAILY_EXCLUSIONS = {
    "simple/hybrid/64": "target message occurs after QA.time",
    "simple/hybrid/179": "target message occurs after QA.time",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
def assert_query_isolation(row: dict[str, Any]) -> None:
    query = row["question"]
    lowered = query.lower()
    forbidden_literals = (
        "supporting_evidence",
        "target_step_id",
        "ground_truth",
        "gold answer",
        "expected answer",
    )
    hits = [value for value in forbidden_literals if value in lowered]
    if row["dataset"].startswith("atm_bench"):
        hits.extend(str(item) for item in row["evidence"] if str(item).lower() in lowered)
    if hits:
        raise ValueError(f"query leakage for {row['instance_id']}: {sorted(set(hits))}")


def atm_memory(raw_dir: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for modality, relative, key in [
        ("image", "data/processed_memory/image_batch_results.json", "image_path"),
        ("video", "data/processed_memory/video_batch_results.json", "video_path"),
        ("email", "data/raw_memory/email/emails.json", "id"),
    ]:
        for source in load_json(raw_dir / relative):
            item_id = str(source[key]) if modality == "email" else Path(str(source[key])).stem
            if modality == "email":
                text = (
                    f"ID: {item_id}\n"
                    f"Timestamp: {source.get('timestamp', '')}\n"
                    f"Summary: {source.get('short_summary', '')}\n"
                    f"Detail: {source.get('detail', '')}\n"
                )
            else:
                tags = source.get("tags", []) or []
                tags_text = ", ".join(str(item) for item in tags) if isinstance(tags, list) else str(tags)
                text = "\n".join(
                    [
                        f"ID: {item_id}",
                        f"Type: {modality}",
                        f"Timestamp: {source.get('timestamp', '')}",
                        f"Location: {source.get('location_name', '')}",
                        f"Short Caption: {source.get('short_caption', '')}",
                        f"Caption: {source.get('caption', '')}",
                        f"OCR: {source.get('ocr_text', '')}",
                        f"Tags: {tags_text}",
                    ]
                ) + "\n"
            if item_id in indexed:
                raise ValueError(f"duplicate ATM evidence id: {item_id}")
            indexed[item_id] = {
                "id": item_id,
                "modality": modality,
                "timestamp": str(source.get("timestamp", "")),
                "content": text,
            }
    return indexed


def atm_prompt(qa: dict[str, Any]) -> str:
    suffix = (
        " If the question asks to recall or list items, respond with the corresponding evidence IDs "
        "only, comma-separated, with no extra text."
        if qa.get("qtype") == "list_recall"
        else ""
    )
    return (
        "You are a QA assistant. Use ONLY the memory written before this query to answer. "
        "If the memory is insufficient, answer 'Unknown'. Respond with only the answer."
        f"{suffix}\nQuestion: {qa['question']}"
    )


def build_atm(raw_dir: Path) -> list[dict[str, Any]]:
    memory = atm_memory(raw_dir)
    qas = load_json(raw_dir / "data/atm-bench/atm-bench.json")
    output = []
    for qa in qas:
        atoms = []
        for evidence_index, raw_id in enumerate(qa.get("evidence_ids", []), 1):
            item_id = Path(str(raw_id)).stem
            item = memory.get(item_id)
            if not item:
                raise ValueError(f"ATM evidence missing for {qa['id']}: {raw_id}")
            atoms.append({**item, "content": f"Evidence {evidence_index}:\n{item['content']}"})
        output.append(
            {
                "task_type": "memqa_ood",
                "dataset": "atm_bench_text_sgm",
                "split": "official_standard_gold_evidence",
                "instance_id": f"atm__{qa['id']}",
                "source_sample_id": qa["id"],
                "context": atoms,
                "memory_steps": [
                    {
                        "step_id": index,
                        "turn_start": item["id"],
                        "turn_end": item["id"],
                        "content": item["content"],
                        "source_turn_count": 1,
                    }
                    for index, item in enumerate(atoms, 1)
                ],
                "question": atm_prompt(qa),
                "answer": qa["answer"],
                "evidence": [item["id"] for item in atoms],
                "metadata": {
                    "source_dataset": "Jingbiao/ATM-Bench",
                    "source_revision": ATM_REVISION,
                    "question_type": qa["qtype"],
                    "raw_category": qa["qtype"],
                    "original_question": qa["question"],
                    "notes": qa.get("notes", ""),
                    "official_evidence_ids": qa.get("evidence_ids", []),
                    "official_evidence_order_preserved": True,
                    "context_setting": "gold_evidence_only",
                    "media_source": "official_batch_results_text_sgm",
                    "selection_rule_version": PROTOCOL_VERSION,
                    "segmentation": "one_official_evidence_item_per_memory_step",
                    "audit_status": "eligible_no_known_overlap",
                    "answerable": True,
                },
            }
        )
    if len(output) != 1013:
        raise ValueError(f"expected 1013 ATM rows, got {len(output)}")
    return output


def flatten_memdaily(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for category in sorted(source):
        for scenario in sorted(source[category]):
            for row in source[category][scenario]:
                rows.append({"category": category, "scenario": scenario, **row})
    return rows


def memdaily_prompt(qa: dict[str, Any]) -> str:
    lines = ["[单选题] " + qa["question"]]
    lines.extend(f"{key}: {value}" for key, value in qa["choices"].items())
    lines.extend(
        [
            f"当前时间是 {qa['time']}",
            "请根据此前写入的[用户消息]，给出[单选题]的正确答案。",
            "只输出答案所对应的选项，不要输出解释或其他任何的内容。",
            "输出样例: A",
        ]
    )
    return "\n".join(lines)


def build_memdaily(source_path: Path) -> list[dict[str, Any]]:
    output = []
    for row in flatten_memdaily(load_json(source_path)):
        row_id = f"{row['category']}/{row['scenario']}/{row['tid']}"
        if row_id in MEMDAILY_EXCLUSIONS:
            continue
        qa = row["QA"]
        target_ids = sorted(set(int(value) for value in qa.get("target_step_id", [])))
        atoms = []
        for target in target_ids:
            if target < 0 or target >= len(row["message_list"]):
                raise ValueError(f"MemDaily target out of range for {row_id}: {target}")
            message = row["message_list"][target]
            if int(message.get("mid", -1)) != target:
                raise ValueError(f"MemDaily mid mismatch for {row_id}: {target}")
            atoms.append(
                {
                    "turn_id": str(target),
                    "timestamp": message.get("time", ""),
                    "place": message.get("place", ""),
                    "content": json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                }
            )
        if qa.get("choices", {}).get(qa.get("ground_truth")) != qa.get("answer"):
            raise ValueError(f"MemDaily choice mismatch for {row_id}")
        output.append(
            {
                "task_type": "memqa_ood",
                "dataset": "memdaily_official",
                "split": "official_release_oracle_target_messages",
                "instance_id": f"memdaily__{row['category']}__{row['scenario']}__{row['tid']}",
                "source_sample_id": row_id,
                "context": atoms,
                "memory_steps": [
                    {
                        "step_id": index,
                        "turn_start": item["turn_id"],
                        "turn_end": item["turn_id"],
                        "content": item["content"],
                        "source_turn_count": 1,
                    }
                    for index, item in enumerate(atoms, 1)
                ],
                "question": memdaily_prompt(qa),
                "answer": qa["ground_truth"],
                "evidence": [item["turn_id"] for item in atoms],
                "metadata": {
                    "source_dataset": "MemDaily official pre-generated release",
                    "source_revision": MEMDAILY_REVISION,
                    "question_type": row["category"],
                    "raw_category": row["category"],
                    "scenario": row["scenario"],
                    "original_question": qa["question"],
                    "answer_text": qa["answer"],
                    "choices": qa["choices"],
                    "question_time": qa["time"],
                    "official_target_step_id": qa["target_step_id"],
                    "context_setting": "gold_evidence_only",
                    "selection_rule_version": PROTOCOL_VERSION,
                    "segmentation": "one_official_target_message_per_memory_step",
                    "audit_status": "eligible_no_known_overlap",
                    "answerable": True,
                },
            }
        )
    if len(output) != 2952:
        raise ValueError(f"expected 2952 MemDaily rows, got {len(output)}")
    return output
