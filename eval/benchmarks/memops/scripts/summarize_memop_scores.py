#!/usr/bin/env python3
"""Summarize scored MemOP JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPORT_EXCLUSION_POLICY = (
    "Default current-reporting policy excludes metis_test operation=mixed rows."
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1 = [float(row.get("score", {}).get("normalized_f1") or 0.0) for row in rows]
    em = [1.0 if row.get("score", {}).get("exact_match") else 0.0 for row in rows]
    judge = [float(row.get("score", {}).get("llm_judge_score") or 0.0) for row in rows]
    strict = [float(row.get("score", {}).get("llm_judge_strict_score") or 0.0) for row in rows]
    return {
        "n": len(rows),
        "f1": avg(f1),
        "em": avg(em),
        "judge": avg(judge),
        "strict_judge": avg(strict),
        "prompt_truncated": sum(1 for row in rows if row.get("prompt_truncated")),
        "audit_issues": sum(len(row.get("audit_issues") or []) for row in rows),
        "runtime_status": dict(Counter(str(row.get("runtime_status") or "ok") for row in rows)),
    }


def key_for(row: dict[str, Any], fields: list[str]) -> str:
    return "|".join(str(row.get(field, "") or "") for field in fields)


def bucket(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_for(row, fields)].append(row)
    return {key: metric_block(part) for key, part in sorted(grouped.items())}


def is_default_report_excluded(row: dict[str, Any]) -> bool:
    return row.get("dataset") == "metis_test" and row.get("operation") == "mixed"


def collect(paths: list[Path], *, include_metis_test_mixed: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for path in paths:
        for row in read_jsonl(path):
            item = dict(row)
            item["_scored_path"] = str(path)
            if not include_metis_test_mixed and is_default_report_excluded(item):
                excluded["metis_test|operation=mixed"] += 1
                continue
            rows.append(item)
    return rows, excluded


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# MemOP Score Summary",
        "",
        f"Records: {summary['overall']['n']}",
        "",
        "## Reporting Policy",
        "",
        summary["exclusion_policy"],
        "",
        f"Excluded rows: `{summary['excluded_rows']}`",
        "",
        "## Overall",
        "",
        "| n | F1 | EM | judge | strict | truncated | audit issues | runtime status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    overall = summary["overall"]
    lines.append(
        f"| {overall['n']} | {overall['f1']:.4f} | {overall['em']:.4f} | "
        f"{overall['judge']:.4f} | {overall['strict_judge']:.4f} | "
        f"{overall['prompt_truncated']} | {overall['audit_issues']} | `{overall['runtime_status']}` |"
    )
    for section in ["by_row", "by_dataset", "by_setting", "by_operation", "by_subtask"]:
        lines.extend(["", f"## {section}", "", "| key | n | F1 | EM | judge | strict |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for key, metrics in summary[section].items():
            lines.append(
                f"| `{key}` | {metrics['n']} | {metrics['f1']:.4f} | {metrics['em']:.4f} | "
                f"{metrics['judge']:.4f} | {metrics['strict_judge']:.4f} |"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--include-metis-test-mixed",
        action="store_true",
        help="Include metis_test operation=mixed rows for historical/all-rows diagnostics.",
    )
    args = parser.parse_args()

    rows, excluded = collect(args.inputs, include_metis_test_mixed=args.include_metis_test_mixed)
    summary = {
        "inputs": [str(path) for path in args.inputs],
        "exclusion_policy": (
            "Historical/all-rows diagnostic: metis_test operation=mixed included."
            if args.include_metis_test_mixed
            else REPORT_EXCLUSION_POLICY
        ),
        "excluded_rows": dict(excluded),
        "overall": metric_block(rows),
        "by_row": bucket(rows, ["baseline", "model_label", "context_policy"]),
        "by_dataset": bucket(rows, ["dataset"]),
        "by_setting": bucket(rows, ["dataset", "setting"]),
        "by_operation": bucket(rows, ["dataset", "setting", "operation"]),
        "by_subtask": bucket(rows, ["dataset", "setting", "subtask"]),
        "context_policies": dict(Counter(str(row.get("context_policy")) for row in rows)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps(summary["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
