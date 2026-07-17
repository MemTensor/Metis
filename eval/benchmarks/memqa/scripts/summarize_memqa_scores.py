#!/usr/bin/env python3
"""Summarize scored MemQA JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.mean(values)), 6)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    f1 = []
    em = []
    judge = []
    strict_judge = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        score = record.get("score", {})
        f1.append(float(score.get("normalized_f1", 0.0)))
        em.append(1.0 if score.get("exact_match") else 0.0)
        llm_score = score.get("llm_judge_score")
        if llm_score is not None:
            judge.append(float(llm_score))
        strict_score = score.get("llm_judge_strict_score")
        if strict_score is not None:
            strict_judge.append(float(strict_score))
        by_category[str(record.get("raw_category"))].append(record)
        source_dataset = record.get("source_dataset")
        if source_dataset is None and record.get("source_sample_id"):
            source_dataset = str(record["source_sample_id"]).split(":", 1)[0]
        by_source_dataset[str(source_dataset)].append(record)
    return {
        "records": len(records),
        "normalized_f1_mean": mean(f1),
        "exact_match_rate": mean(em),
        "llm_judge_mean": mean(judge),
        "llm_judge_count": len(judge),
        "llm_judge_strict_mean": mean(strict_judge),
        "llm_judge_strict_count": len(strict_judge),
        "by_raw_category": {category: summarize_records(items) for category, items in sorted(by_category.items())}
        if len(by_category) > 1
        else {},
        "by_source_dataset": {source: summarize_records(items) for source, items in sorted(by_source_dataset.items())}
        if len(by_source_dataset) > 1
        else {},
    }


def flatten_summary(paths: list[Path]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = {}
    for path in paths:
        records = read_jsonl(path)
        files[str(path)] = len(records)
        for record in records:
            key = "|".join(
                [
                    str(record.get("baseline")),
                    str(record.get("model_label")),
                    str(record.get("context_policy")),
                ]
            )
            groups[key].append(record)
    return {
        "files": files,
        "groups": {
            key: summarize_records(records)
            for key, records in sorted(groups.items())
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MemQA Score Summary",
        "",
        "## Files",
        "",
        "| file | records |",
        "| --- | ---: |",
    ]
    for path, count in summary["files"].items():
        lines.append(f"| `{path}` | {count} |")
    lines.extend(
        [
            "",
            "## Overall By Baseline / Model / Context",
            "",
            "| key | records | F1 | EM | judge | judge N | judge_strict | strict N |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, data in summary["groups"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                key,
                data["records"],
                data["normalized_f1_mean"],
                data["exact_match_rate"],
                data["llm_judge_mean"],
                data["llm_judge_count"],
                data["llm_judge_strict_mean"],
                data["llm_judge_strict_count"],
            )
        )
    lines.extend(["", "## Category Breakdowns", ""])
    for key, data in summary["groups"].items():
        if not data.get("by_raw_category"):
            continue
        lines.extend(
            [
                f"### `{key}`",
                "",
                "| raw_category | records | F1 | EM | judge | judge N | judge_strict | strict N |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for category, cat_data in data["by_raw_category"].items():
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    category,
                    cat_data["records"],
                    cat_data["normalized_f1_mean"],
                    cat_data["exact_match_rate"],
                    cat_data["llm_judge_mean"],
                    cat_data["llm_judge_count"],
                    cat_data["llm_judge_strict_mean"],
                    cat_data["llm_judge_strict_count"],
                )
            )
        lines.append("")
    lines.extend(["", "## Source Dataset Breakdowns", ""])
    for key, data in summary["groups"].items():
        if not data.get("by_source_dataset"):
            continue
        lines.extend(
            [
                f"### `{key}`",
                "",
                "| source_dataset | records | F1 | EM | judge | judge N | judge_strict | strict N |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source_dataset, source_data in data["by_source_dataset"].items():
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    source_dataset,
                    source_data["records"],
                    source_data["normalized_f1_mean"],
                    source_data["exact_match_rate"],
                    source_data["llm_judge_mean"],
                    source_data["llm_judge_count"],
                    source_data["llm_judge_strict_mean"],
                    source_data["llm_judge_strict_count"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = flatten_summary(args.inputs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
