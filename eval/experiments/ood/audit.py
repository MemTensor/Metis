#!/usr/bin/env python3
"""Strictly audit every current OOD cell from the tracked protocol."""

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


DEFAULT_CONFIG = EVAL_ROOT / "configs/paper/table_8_ood.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temp_lora_runtime_failures(metadata: dict[str, Any], method: dict[str, Any]) -> list[str]:
    runtime = (metadata.get("runtime") or {}).get("runtime_summary") or {}
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


def audit_cell(
    run_dir: Path,
    data_dir: Path,
    method: dict[str, Any],
    dataset: str,
    expected: int,
    judge: dict[str, Any],
) -> dict[str, Any]:
    cell = run_dir / "cells" / method["id"] / dataset
    raw = cell / "raw.jsonl"
    raw_meta = raw.with_suffix(".meta.json")
    scored = cell / "scored.jsonl"
    score_meta = scored.with_suffix(".score_meta.json")
    required = [raw, raw_meta, scored, score_meta, cell / "summary.json"]
    failures = [f"missing:{path.name}" for path in required if not path.is_file()]
    if failures:
        return {"cell": f"{method['id']}__{dataset}", "status": "fail", "failures": failures}

    item = dataset_map()[dataset]
    instances = read_jsonl(data_dir / item["path"])
    raw_rows = read_jsonl(raw)
    scored_rows = read_jsonl(scored)
    instance_ids = [row["instance_id"] for row in instances]
    raw_ids = [row["instance_id"] for row in raw_rows]
    scored_ids = [row["instance_id"] for row in scored_rows]
    if not (len(instances) == len(raw_rows) == len(scored_rows) == expected):
        failures.append("row_count")
    if raw_ids != instance_ids or scored_ids != instance_ids:
        failures.append("id_order_or_coverage")
    if len(raw_ids) != len(set(raw_ids)) or len(scored_ids) != len(set(scored_ids)):
        failures.append("duplicate_ids")
    if any(row.get("audit_issues") for row in raw_rows):
        failures.append("query_audit")
    if any(
        not math.isfinite(float(row[key]))
        for row in raw_rows
        for key in ("latency_sec", "query_latency_sec")
        if key in row
    ):
        failures.append("runtime_nonfinite")
    metadata = load(raw_meta)
    if method["implementation"] == "metis" and metadata.get("runtime", {}).get("load_report", {}).get("important_missing"):
        failures.append("metis_load_report")
    if method["implementation"] == "temp_lora":
        failures.extend(temp_lora_runtime_failures(metadata, method))
    if metadata.get("resume_state", {}).get("status") != "complete":
        failures.append("resume_state")
    scored_metadata = load(score_meta)
    if scored_metadata.get("judge_failure_count") != 0:
        failures.append("judge_failures")
    if not scored_metadata.get("strict_only") or scored_metadata.get("judge_repeats") != judge["repeats"]:
        failures.append("score_metadata_protocol")
    if scored_metadata.get("judge_model") != judge["model"]:
        failures.append("judge_model")
    if float(scored_metadata.get("judge_temperature", -1)) != float(judge["temperature"]):
        failures.append("judge_temperature")
    if (scored_metadata.get("judge_api_status") or {}).get("available") is not True:
        failures.append("judge_api_unavailable")
    scorer = EVAL_ROOT / "benchmarks/ood/scripts/score_official_gold.py"
    if scored_metadata.get("scorer_code_sha256") != sha256(scorer):
        failures.append("scorer_code_sha256")
    if (scored_metadata.get("official_code") or {}).get("atm_revision") != "d463445614ad78a48736b98ab901795f7ecaf3da":
        failures.append("atm_revision")
    return {
        "cell": f"{method['id']}__{dataset}",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "rows": len(scored_rows),
        "primary_score": load(cell / "summary.json").get("primary_score"),
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
    expected = {key: value["rows"] for key, value in dataset_map().items() if key in config["datasets"]}
    audits = [
        audit_cell(args.run_dir, args.data_dir, method, dataset, expected[dataset], config["judge"])
        for method in config["methods"]
        for dataset in config["datasets"]
    ]
    failures = [item for item in audits if item["status"] != "pass"]
    dataset_index = {item: index for index, item in enumerate(reported["ood"]["datasets"])}
    for cell in audits:
        method_id, dataset = cell["cell"].split("__", 1)
        expected_score = reported["ood"]["scores"][method_id][dataset_index[dataset]]
        cell["paper_score_pp"] = expected_score
        cell["delta_pp"] = None if cell.get("primary_score") is None else cell["primary_score"] * 100 - expected_score
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
    data_report = verify(args.data_dir, only=set(config["datasets"]))
    report = {
        "status": "pass" if not failures and not asset_failures and data_report["ok"] else "fail",
        "cell_count": len(audits),
        "data": data_report,
        "model_assets": model_assets,
        "cells": audits,
    }
    write_json(args.run_dir / "audit.json", report)
    if failures or asset_failures or not data_report["ok"]:
        raise RuntimeError(f"{len(failures)} of {len(audits)} OOD cells failed audit")
    print(json.dumps({"status": "pass", "cells": len(audits)}))


if __name__ == "__main__":
    main()
