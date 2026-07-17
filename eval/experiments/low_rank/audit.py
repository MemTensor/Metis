#!/usr/bin/env python3
"""Portable strict audit for the paper's 28-cell LowRankMemory workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from eval.common.assets import DEFAULT_ASSET_REGISTRY, dataset_map, load_assets, resolve_asset
from eval.common.cli import read_jsonl, write_json
from eval.common.paths import EVAL_ROOT
from eval.data.verify import verify


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_10_low_rank.json"
FORBIDDEN_QUERY_FIELDS = {"supporting_evidence", "target_step_id", "ground_truth", "answer_text"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_label(rank: int | str) -> str:
    return "rankfull" if rank == "full" else f"rank{int(rank):04d}"


def validate_low_rank_config(meta: dict[str, Any], rank: int | str) -> list[str]:
    config = meta.get("low_rank_local_memory") or {}
    if rank == "full":
        return [] if config.get("enabled") is False else ["full_rank_projection_enabled"]
    issues = []
    expected = {"enabled": True, "rank": int(rank), "policy": "after_each_commit", "target": "state"}
    for key, value in expected.items():
        if config.get(key) != value:
            issues.append(f"low_rank_config:{key}")
    return issues


def audit_cell(
    run_dir: Path,
    data_dir: Path,
    model: dict[str, Any],
    benchmark: dict[str, Any],
    rank: int | str,
    manifest: dict[str, dict[str, Any]],
    judge: dict[str, Any],
) -> dict[str, Any]:
    prefix = model["model_label_prefix"]
    label = f"{prefix}_{rank_label(rank)}"
    model_dir = run_dir / "models" / model["id"]
    raw = model_dir / "raw" / benchmark["runner_name"] / f"{label}.raw.jsonl"
    scored = model_dir / "scored_strict" / benchmark["runner_name"] / f"{label}.scored.jsonl"
    raw_meta = raw.with_suffix(".meta.json")
    score_meta = scored.with_suffix(".score_meta.json")
    files = [raw, raw_meta, scored, score_meta]
    failures = [f"missing:{path.name}" for path in files if not path.is_file()]
    cell_id = f"{model['id']}__{benchmark['id']}__{rank_label(rank)}"
    if failures:
        return {"cell": cell_id, "status": "fail", "failures": failures}

    instances = read_jsonl(data_dir / manifest[benchmark["id"]]["path"])
    raw_rows = read_jsonl(raw)
    scored_rows = read_jsonl(scored)
    expected_rows = manifest[benchmark["id"]]["rows"]
    instance_ids = [row["instance_id"] for row in instances]
    raw_ids = [row["instance_id"] for row in raw_rows]
    scored_ids = [row["instance_id"] for row in scored_rows]
    if not (len(instances) == len(raw_rows) == len(scored_rows) == expected_rows):
        failures.append("row_count")
    if raw_ids != instance_ids or scored_ids != instance_ids:
        failures.append("id_order_or_coverage")
    if len(raw_ids) != len(set(raw_ids)) or len(scored_ids) != len(set(scored_ids)):
        failures.append("duplicate_ids")
    if any(row.get("audit_issues") for row in raw_rows):
        failures.append("query_audit")
    if any(row.get("runtime_status", "ok") != "ok" for row in raw_rows):
        failures.append("runtime_status")
    if any(row.get("context_policy") not in {"metis_memory_only", "metis_memory_state_only"} for row in raw_rows):
        failures.append("context_policy")
    if any(not math.isfinite(float(row[key])) for row in raw_rows for key in ("latency_sec", "query_latency_sec") if key in row):
        failures.append("runtime_nonfinite")
    if any(key in json.dumps(row.get("query_payload", "")).lower() for row in raw_rows for key in FORBIDDEN_QUERY_FIELDS):
        failures.append("forbidden_query_field")

    metadata = load(raw_meta)
    failures.extend(validate_low_rank_config(metadata, rank))
    report = metadata.get("load_report") or {}
    if report.get("important_missing"):
        failures.append("important_missing")
    if report.get("unexpected_keys") or report.get("unexpected"):
        failures.append("unexpected_checkpoint_keys")
    if report.get("checkpoint_format") != "delta":
        failures.append("checkpoint_format")
    if rank != "full":
        debug_rows = [row.get("method_debug", {}).get("low_rank_local_memory") for row in raw_rows]
        if any(not isinstance(item, dict) for item in debug_rows):
            failures.append("numeric_rank_debug_missing")
        elif any(item.get("config", {}).get("rank") not in {None, int(rank)} for item in debug_rows):
            failures.append("numeric_rank_debug_mismatch")

    strict_values = [(row.get("score") or {}).get("llm_judge_strict_score") for row in scored_rows]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in strict_values):
        failures.append("strict_score_non_numeric")
    sources = [(row.get("score") or {}).get("strict_judge", {}).get("judge_source") for row in scored_rows]
    if any(source != "api_median" for source in sources):
        failures.append("strict_score_source")
    score_metadata = load(score_meta)
    if not score_metadata.get("strict_only") or score_metadata.get("judge_repeats") != 3:
        failures.append("score_metadata_protocol")
    judge_status = score_metadata.get("judge_api_status") or {}
    if judge_status.get("requested_model") != judge["judge_model"] or judge_status.get("selected_model") != judge["judge_model"]:
        failures.append("judge_model")
    if judge_status.get("available") is not True:
        failures.append("judge_api_unavailable")
    if score_metadata.get("judge_failure_count", 0) != 0:
        failures.append("judge_failures")
    if any(source in {None, "api_error", "unavailable"} for source in sources):
        failures.append("judge_status")
    if benchmark["id"] == "metis_test_nomixed":
        observed = Counter(row.get("operation") for row in instances)
        if observed != Counter({"remember": 212, "update": 240, "forget": 240, "reflect": 160}):
            failures.append("metis_test_operation_counts")
    return {
        "cell": cell_id,
        "status": "pass" if not failures else "fail",
        "failures": sorted(set(failures)),
        "rows": len(scored_rows),
        "strict_mean": sum(float(value) for value in strict_values if isinstance(value, (int, float))) / len(strict_values) if strict_values else None,
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
    manifest = dataset_map()
    assets = load_assets(args.assets)
    model_assets = []
    for model in config["models"]:
        checkpoint = Path(resolve_asset(model["checkpoint"], assets))
        delta = checkpoint / "metis_delta.safetensors"
        failures = []
        if not delta.is_file():
            failures.append("missing_delta")
        elif sha256(delta) != model["delta_sha256"]:
            failures.append("delta_sha256")
        model_assets.append({"model": model["id"], "checkpoint": str(checkpoint), "status": "pass" if not failures else "fail", "failures": failures})
    cells = [
        audit_cell(args.run_dir, args.data_dir, model, benchmark, rank, manifest, config["scoring"])
        for model in config["models"]
        for benchmark in config["benchmarks"]
        for rank in config["ranks"]
    ]
    rank_index = {str(item): index for index, item in enumerate(reported["low_rank"]["ranks"])}
    for cell in cells:
        _, benchmark_id, observed_rank = cell["cell"].split("__", 2)
        rank = "full" if observed_rank == "rankfull" else str(int(observed_rank.removeprefix("rank")))
        expected = reported["low_rank"]["scores"][benchmark_id][rank_index[rank]]
        cell["paper_score_pp"] = expected
        cell["delta_pp"] = None if cell.get("strict_mean") is None else cell["strict_mean"] * 100 - expected
    failures = [item for item in cells if item["status"] != "pass"]
    asset_failures = [item for item in model_assets if item["status"] != "pass"]
    data_report = verify(args.data_dir, only={item["id"] for item in config["benchmarks"]})
    report = {
        "status": "pass" if not failures and not asset_failures and data_report["ok"] else "fail",
        "protocol_status": config["protocol_status"],
        "result_status": config["result_status"],
        "cell_count": len(cells),
        "data": data_report,
        "model_assets": model_assets,
        "cells": cells,
    }
    write_json(args.run_dir / "audit.json", report)
    if failures or asset_failures or not data_report["ok"]:
        raise RuntimeError(f"LowRankMemory audit failed: {len(failures)} cells, {len(asset_failures)} model assets")
    print(json.dumps({"status": "pass", "cells": len(cells)}))


if __name__ == "__main__":
    main()
