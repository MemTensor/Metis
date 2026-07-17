#!/usr/bin/env python3
"""Strictly audit all 77 main-table cells against the tracked protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eval.common.assets import DEFAULT_ASSET_REGISTRY, dataset_map, load_assets, resolve_asset
from eval.common.cli import read_jsonl, write_json
from eval.common.paths import EVAL_ROOT
from eval.data.verify import verify
from eval.experiments.main_tables.run import (
    expected_max_input_tokens,
    expected_max_new_tokens,
)


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_5_6_main.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temp_lora_runtime_failures(metadata: dict[str, Any], method: dict[str, Any]) -> list[str]:
    runtime = metadata.get("runtime_summary") or metadata.get("temp_lora") or {}
    failures = []
    if runtime.get("seed") != method.get("seed"):
        failures.append("temp_lora_seed")
    if method["id"] != "temp_lora27b":
        return failures
    if runtime.get("temp_lora_device_map") != "balanced":
        failures.append("temp_lora_27b_device_map")
    if runtime.get("temp_lora_max_memory") != ["0:76GiB", "1:76GiB"]:
        failures.append("temp_lora_27b_max_memory")
    devices = {
        str(value)
        for value in (runtime.get("hf_device_map") or {}).values()
        if str(value) not in {"cpu", "disk", "meta"}
    }
    if len(devices) < 2:
        failures.append("temp_lora_27b_not_sharded")
    return failures


def qwen27_runtime_failures(metadata: dict[str, Any], method: dict[str, Any]) -> list[str]:
    if method["id"] not in {
        "qwen27b_no_context",
        "qwen27b_full_context",
        "dense_rag27b",
    }:
        return []
    failures = []
    if metadata.get("device_map") != "auto":
        failures.append("qwen27b_device_map")
    if metadata.get("max_memory") != ["0:75GiB", "1:75GiB"]:
        failures.append("qwen27b_max_memory")
    devices = {
        str(value)
        for value in (metadata.get("hf_device_map") or {}).values()
        if str(value) not in {"cpu", "disk", "meta"}
    }
    if len(devices) < 2:
        failures.append("qwen27b_not_sharded")
    return failures


def cell_files(run_dir: Path, method: dict[str, Any], benchmark: dict[str, Any]) -> tuple[Path, Path]:
    cell = run_dir / "cells" / f"{method['id']}__{benchmark['id']}"
    implementation = method["implementation"]
    if implementation in {"no_context", "full_context"}:
        raw = cell / f"{method['id']}.{implementation}.raw.jsonl"
    else:
        raw = cell / f"{benchmark['runner_id']}.{implementation}.raw.jsonl"
    scored = cell / f"{benchmark['runner_id']}.{implementation}.scored.jsonl"
    return raw, scored


def audit_cell(
    run_dir: Path,
    data_dir: Path,
    config: dict[str, Any],
    method: dict[str, Any],
    benchmark: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    judge: dict[str, Any],
) -> dict[str, Any]:
    cell_id = f"{method['id']}__{benchmark['id']}"
    raw, scored = cell_files(run_dir, method, benchmark)
    raw_meta = raw.with_suffix(".meta.json")
    score_meta = scored.with_suffix(".score_meta.json")
    required = [raw, raw_meta, scored, score_meta]
    failures = [f"missing:{path.name}" for path in required if not path.is_file()]
    if failures:
        return {"cell": cell_id, "status": "fail", "failures": failures}

    instances = read_jsonl(data_dir / manifest[benchmark["id"]]["path"])
    raw_rows = read_jsonl(raw)
    scored_rows = read_jsonl(scored)
    expected = manifest[benchmark["id"]]["rows"]
    ids = [row["instance_id"] for row in instances]
    raw_ids = [row["instance_id"] for row in raw_rows]
    scored_ids = [row["instance_id"] for row in scored_rows]
    if not (len(instances) == len(raw_rows) == len(scored_rows) == expected):
        failures.append("row_count")
    if raw_ids != ids or scored_ids != ids:
        failures.append("id_order_or_coverage")
    if len(raw_ids) != len(set(raw_ids)) or len(scored_ids) != len(set(scored_ids)):
        failures.append("duplicate_ids")
    if any(
        any(scored_row.get(key) != value for key, value in raw_row.items())
        for raw_row, scored_row in zip(raw_rows, scored_rows)
    ):
        failures.append("scored_raw_mismatch")
    if method["implementation"] in {"delta_mem", "temp_lora", "metis"} and any(row.get("audit_issues") for row in raw_rows):
        failures.append("query_audit")
    if any(row.get("runtime_status", "ok") != "ok" for row in raw_rows):
        failures.append("runtime_status")
    if any(not math.isfinite(float(row[key])) for row in raw_rows for key in ("latency_sec", "query_latency_sec") if key in row):
        failures.append("runtime_nonfinite")

    metadata = load(raw_meta)
    failures.extend(qwen27_runtime_failures(metadata, method))
    expected_generation_tokens = expected_max_new_tokens(
        config, method["implementation"], benchmark["runner_id"]
    )
    observed_generation_tokens = {
        (row.get("generation_config") or {}).get("max_new_tokens")
        for row in raw_rows
    }
    if observed_generation_tokens != {expected_generation_tokens}:
        failures.append("max_new_tokens_protocol")
    expected_input_tokens = expected_max_input_tokens(
        config, method["implementation"], benchmark["runner_id"]
    )
    if expected_input_tokens and metadata.get("max_input_tokens") != expected_input_tokens:
        failures.append("max_input_tokens_protocol")
    if method["implementation"] == "metis" and (metadata.get("load_report") or {}).get("important_missing"):
        failures.append("metis_load_report")
    if method["implementation"] == "temp_lora":
        failures.extend(temp_lora_runtime_failures(metadata, method))
    strict_values = [(row.get("score") or {}).get("llm_judge_strict_score") for row in scored_rows]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in strict_values):
        failures.append("strict_score_non_numeric")
    sources = [(row.get("score") or {}).get("strict_judge", {}).get("judge_source") for row in scored_rows]
    if any(source != "api_median" for source in sources):
        failures.append("strict_score_source")
    scored_metadata = load(score_meta)
    if not scored_metadata.get("strict_only") or scored_metadata.get("judge_repeats") != 3:
        failures.append("score_metadata_protocol")
    judge_status = scored_metadata.get("judge_api_status") or {}
    if judge_status.get("requested_model") != judge["model"] or judge_status.get("selected_model") != judge["model"]:
        failures.append("judge_model")
    if float(judge_status.get("judge_temperature", -1)) != float(judge["temperature"]):
        failures.append("judge_temperature")
    if judge_status.get("available") is not True:
        failures.append("judge_api_unavailable")
    if scored_metadata.get("judge_failure_count", 0) != 0:
        failures.append("judge_failures")
    if any(source in {None, "api_error", "unavailable"} for source in sources):
        failures.append("judge_status")
    return {
        "cell": cell_id,
        "status": "pass" if not failures else "fail",
        "failures": sorted(set(failures)),
        "rows": len(scored_rows),
        "strict_mean": sum(float(value) for value in strict_values) / len(strict_values) if strict_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_REGISTRY)
    args = parser.parse_args()
    config = load(args.config)
    reported = load(EVAL_ROOT / "configs/paper/reported_scores.json")
    assets = load_assets(args.assets)
    manifest = dataset_map()
    cells = [
        audit_cell(
            args.run_dir,
            args.data_dir,
            config,
            method,
            benchmark,
            manifest,
            config["judge"],
        )
        for method in config["methods"]
        for benchmark in config["benchmarks"]
        if benchmark["id"] not in set(method.get("skip_benchmarks", []))
    ]
    failures = [item for item in cells if item["status"] != "pass"]
    benchmark_index = {item: index for index, item in enumerate(reported["main"]["benchmarks"])}
    for cell in cells:
        method_id, benchmark_id = cell["cell"].split("__", 1)
        expected = reported["main"]["scores"][method_id][benchmark_index[benchmark_id]]
        cell["paper_score_pp"] = expected
        cell["delta_pp"] = None if cell.get("strict_mean") is None or expected is None else cell["strict_mean"] * 100 - expected
    model_assets = []
    for method in config["methods"]:
        if method["implementation"] != "metis":
            continue
        checkpoint_id = method["checkpoint"]
        delta = Path(resolve_asset(checkpoint_id, assets)) / "metis_delta.safetensors"
        expected_hash = reported["metis_checkpoints"][checkpoint_id]["delta_sha256"]
        asset_failures = []
        if not delta.is_file():
            asset_failures.append("missing_delta")
        elif sha256(delta) != expected_hash:
            asset_failures.append("delta_sha256")
        model_assets.append({
            "method": method["id"],
            "checkpoint": checkpoint_id,
            "status": "pass" if not asset_failures else "fail",
            "failures": asset_failures,
        })
    asset_failures = [item for item in model_assets if item["status"] != "pass"]
    data_report = verify(args.data_dir, only={item["id"] for item in config["benchmarks"]})
    report = {
        "status": "pass" if not failures and not asset_failures and data_report["ok"] else "fail",
        "cell_count": len(cells),
        "data": data_report,
        "model_assets": model_assets,
        "cells": cells,
    }
    write_json(args.run_dir / "audit.json", report)
    if len(cells) != config["expected_cells"]:
        raise RuntimeError(f"audit expanded {len(cells)} cells, expected {config['expected_cells']}")
    if failures or asset_failures or not data_report["ok"]:
        raise RuntimeError(f"{len(failures)} of {len(cells)} main-table cells failed audit")
    print(json.dumps({"status": "pass", "cells": len(cells)}))


if __name__ == "__main__":
    main()
