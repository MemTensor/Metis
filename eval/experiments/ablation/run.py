#!/usr/bin/env python3
"""Audit and run the Metis data/structure ablation matrices."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from eval.common.assets import DEFAULT_ASSET_REGISTRY, load_assets, resolve_asset


REPO_ROOT = Path(__file__).resolve().parents[3]
METIS_DEV_ROOT = Path(os.environ.get("METIS_MODEL_REPO_ROOT", str(REPO_ROOT)))
DEFAULT_MATRIX = Path(__file__).resolve().parent / "configs" / "ablation_matrix.json"
READY_STATUSES = {"ready", "reference_ready"}
REQUIRED_CHECKPOINT_FILES = {
    "config.json",
    "metis_delta.safetensors",
    "metis_delta_manifest.json",
    "tokenizer.json",
    "trainer_state.json",
    "training_info.json",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def jsonl_scan(path: Path) -> dict[str, Any]:
    count = 0
    first_id = None
    last_id = None
    audit_issue_rows = 0
    runtime_non_ok_rows = 0
    strict_values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            first_id = first_id or instance_id
            last_id = instance_id
            count += 1
            if row.get("audit_issues"):
                audit_issue_rows += 1
            if str(row.get("runtime_status") or "ok") != "ok":
                runtime_non_ok_rows += 1
            strict = (row.get("score") or {}).get("llm_judge_strict_score")
            if strict is not None:
                strict_values.append(float(strict))
    return {
        "rows": count,
        "first_instance_id": first_id,
        "last_instance_id": last_id,
        "audit_issue_rows": audit_issue_rows,
        "runtime_non_ok_rows": runtime_non_ok_rows,
        "strict_mean": sum(strict_values) / len(strict_values) if strict_values else None,
        "strict_values": len(strict_values),
    }


def git_snapshot(root: Path = REPO_ROOT) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, check=True, text=True, capture_output=True
    ).stdout
    return {"root": str(root), "head": head, "status_short": status}


def load_dotenv(env: dict[str, str]) -> dict[str, str]:
    """Load the repo-local judge environment without logging secret values."""

    dotenv = REPO_ROOT / ".env"
    if not dotenv.is_file():
        return env
    loaded = dict(env)
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded.setdefault(key, value)
    return loaded


def matrix_maps(matrix: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = {item["id"]: item for item in matrix["checkpoints"]}
    benchmarks = {item["id"]: item for item in matrix["benchmarks"]}
    return checkpoints, benchmarks


def selected_benchmarks(matrix: dict[str, Any], include_appendix: bool) -> list[dict[str, Any]]:
    _, benchmark_map = matrix_maps(matrix)
    main_ids = matrix.get("main_text_benchmark_ids")
    if not main_ids:
        return list(matrix["benchmarks"])
    selected_ids = list(main_ids)
    if include_appendix:
        selected_ids.extend(matrix.get("appendix_benchmark_ids") or [])
    unknown = [benchmark_id for benchmark_id in selected_ids if benchmark_id not in benchmark_map]
    if unknown:
        raise ValueError(f"Unknown benchmark ids in reporting policy: {unknown}")
    return [benchmark_map[benchmark_id] for benchmark_id in selected_ids]


def audit_benchmark(item: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / item["instances"]
    result: dict[str, Any] = {"id": item["id"], "path": str(path), "ok": True, "issues": []}
    if not path.is_file():
        result["ok"] = False
        result["issues"].append("missing instances file")
        return result
    observed = jsonl_scan(path)
    result["observed"] = observed
    result["sha256"] = sha256(path)
    checks = {
        "rows": observed["rows"] == item["rows"],
        "first_instance_id": observed["first_instance_id"] == item["first_instance_id"],
        "last_instance_id": observed["last_instance_id"] == item["last_instance_id"],
        "sha256": result["sha256"] == item["sha256"],
    }
    result["checks"] = checks
    result["ok"] = all(checks.values())
    result["issues"].extend(key for key, value in checks.items() if not value)
    return result


def audit_checkpoint(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["checkpoint"])
    required = item["status"] in READY_STATUSES
    result: dict[str, Any] = {
        "id": item["id"],
        "status": item["status"],
        "path": str(path),
        "required_ready": required,
        "ok": True,
        "issues": [],
    }
    if not path.is_dir():
        result["present"] = False
        if required:
            result["ok"] = False
            result["issues"].append("missing ready checkpoint directory")
        return result
    result["present"] = True
    files = {child.name for child in path.iterdir() if child.is_file()}
    missing = sorted(REQUIRED_CHECKPOINT_FILES - files)
    if missing:
        result["issues"].append(f"missing files: {missing}")

    try:
        config = load_json(path / "config.json")
        state = load_json(path / "trainer_state.json")
        manifest = load_json(path / "metis_delta_manifest.json")
        memory = config.get("memory_configs") or {}
        observed = {
            "step": state.get("global_step"),
            "epoch": state.get("epoch"),
            "base_model_path": manifest.get("base_model_path"),
            "delta_tensors": manifest.get("num_tensors"),
            "hyper_memory_type": memory.get("metis_hyper_memory_type"),
            "local_memory_type": memory.get("metis_local_memory_type"),
            "block_type": memory.get("metis_block_type"),
            "delta_sha256": sha256(path / "metis_delta.safetensors"),
            "config_sha256": sha256(path / "config.json"),
            "files": len(files),
        }
        result["observed"] = observed
        expected_checks = {
            "step": observed["step"] == item["step"],
            "epoch": abs(float(observed["epoch"]) - float(item["epoch"])) < 1e-9,
            "delta_tensors": observed["delta_tensors"] == item.get("expected_delta_tensors"),
            "hyper_memory_type": observed["hyper_memory_type"] == item.get("expected_hyper_memory_type"),
            "local_memory_type": observed["local_memory_type"] == item.get("expected_local_memory_type"),
            "block_type": observed["block_type"] == item.get("expected_block_type"),
        }
        if item.get("delta_sha256"):
            expected_checks["delta_sha256"] = observed["delta_sha256"] == item["delta_sha256"]
        if item.get("config_sha256"):
            expected_checks["config_sha256"] = observed["config_sha256"] == item["config_sha256"]
        if item.get("expected_manifest_base_model_path"):
            expected_checks["base_model_path"] = (
                observed["base_model_path"] == item["expected_manifest_base_model_path"]
            )
        result["checks"] = expected_checks
        result["issues"].extend(key for key, value in expected_checks.items() if not value)
    except Exception as exc:  # audit must report a precise failure instead of hiding it
        result["issues"].append(f"metadata audit failed: {exc.__class__.__name__}: {exc}")

    result["ok"] = not result["issues"]
    return result


def run_static_audit(
    matrix: dict[str, Any], matrix_path: Path, output: Path | None = None
) -> dict[str, Any]:
    report = {
        "created_at": utc_now(),
        "matrix": display_path(matrix_path),
        "matrix_sha256": sha256(matrix_path),
        "git": git_snapshot(),
        "benchmarks": [audit_benchmark(item) for item in matrix["benchmarks"]],
        "checkpoints": [audit_checkpoint(item) for item in matrix["checkpoints"]],
    }
    report["ready_ok"] = all(item["ok"] for item in report["benchmarks"]) and all(
        item["ok"] for item in report["checkpoints"] if item["required_ready"]
    )
    if output:
        write_json(output, report)
    return report


def parse_gpu_map(values: Iterable[str], fallback_gpu: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --gpu-map {value!r}; expected CHECKPOINT_ID=GPU")
        checkpoint_id, gpu = value.split("=", 1)
        if not checkpoint_id or not gpu:
            raise ValueError(f"Invalid --gpu-map {value!r}")
        mapping[checkpoint_id] = gpu
    if fallback_gpu is not None:
        mapping["*"] = fallback_gpu
    return mapping


def parse_reuse_scored(values: Iterable[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --reuse-scored {value!r}; expected CHECKPOINT_ID=RUN_DIR"
            )
        checkpoint_id, run_dir = value.split("=", 1)
        if not checkpoint_id or not run_dir:
            raise ValueError(
                f"Invalid --reuse-scored {value!r}; expected CHECKPOINT_ID=RUN_DIR"
            )
        if checkpoint_id in mapping:
            raise ValueError(f"Duplicate --reuse-scored checkpoint: {checkpoint_id}")
        mapping[checkpoint_id] = Path(run_dir).resolve()
    return mapping


def selected_checkpoints(
    matrix: dict[str, Any],
    axis: str,
    explicit_ids: list[str],
    include_pending: bool,
    mode: str,
    rerun_reference: bool,
) -> list[dict[str, Any]]:
    checkpoints, _ = matrix_maps(matrix)
    if explicit_ids:
        unknown = sorted(set(explicit_ids) - set(checkpoints))
        if unknown:
            raise ValueError(f"Unknown checkpoint ids: {unknown}")
        selected = [checkpoints[item] for item in explicit_ids]
    else:
        selected = [
            item
            for item in matrix["checkpoints"]
            if axis == "all" and item.get("axes") or axis in item.get("axes", [])
        ]
    selected = [item for item in selected if item["status"] != "excluded_confounded"]
    if not include_pending:
        selected = [item for item in selected if item["status"] in READY_STATUSES]
    if mode == "full" and not rerun_reference:
        selected = [item for item in selected if item["status"] != "reference_ready"]
    return selected


def expected_rows(benchmark: dict[str, Any], mode: str) -> int:
    return 1 if mode == "smoke" else int(benchmark["rows"])


def raw_path(run_dir: Path, checkpoint_id: str, benchmark_id: str) -> Path:
    return run_dir / "raw" / checkpoint_id / f"{benchmark_id}.raw.jsonl"


def scored_path(run_dir: Path, checkpoint_id: str, benchmark_id: str) -> Path:
    return run_dir / "scored_strict_r3" / checkpoint_id / f"{benchmark_id}.scored.jsonl"


def raw_complete(path: Path, benchmark: dict[str, Any], mode: str) -> bool:
    if not path.is_file():
        return False
    observed = jsonl_scan(path)
    if observed["rows"] != expected_rows(benchmark, mode):
        return False
    if observed["first_instance_id"] != benchmark["first_instance_id"]:
        return False
    if mode == "full" and observed["last_instance_id"] != benchmark["last_instance_id"]:
        return False
    return observed["audit_issue_rows"] == 0 and observed["runtime_non_ok_rows"] == 0


def raw_meta_complete(path: Path, checkpoint: dict[str, Any]) -> bool:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return False
    meta = load_json(meta_path)
    report = meta.get("load_report") or {}
    recorded_checkpoint = meta.get("model_path") or meta.get("checkpoint")
    expected_base_model = checkpoint.get("expected_base_model_id", "Qwen/Qwen3.5-4B")
    observed_base_model = str(report.get("base_model_path") or "")
    base_model_matches = bool(
        observed_base_model
        and (
            observed_base_model == expected_base_model
            or Path(observed_base_model).name == Path(expected_base_model).name
        )
    )
    expected_model_family = checkpoint.get("expected_model_family")
    return bool(
        recorded_checkpoint
        and Path(recorded_checkpoint).resolve() == Path(checkpoint["checkpoint"]).resolve()
        and report.get("checkpoint_format") == "delta"
        and report.get("state_keys") == checkpoint["expected_delta_tensors"]
        and report.get("delta_manifest_tensor_count") == checkpoint["expected_delta_tensors"]
        and base_model_matches
        and (expected_model_family is None or report.get("model_family") == expected_model_family)
        and not report.get("important_missing")
        and not report.get("unexpected")
        and not report.get("delta_manifest_missing")
        and not report.get("delta_manifest_extra")
    )


def scored_complete(path: Path, benchmark: dict[str, Any], mode: str, evaluation: dict[str, Any]) -> bool:
    meta_path = path.with_suffix(".score_meta.json")
    if not path.is_file() or not meta_path.is_file():
        return False
    observed = jsonl_scan(path)
    meta = load_json(meta_path)
    expected = expected_rows(benchmark, mode)
    status = meta.get("judge_api_status") or {}
    return bool(
        observed["rows"] == expected
        and observed["strict_values"] == expected
        and meta.get("records") == expected
        and meta.get("strict_judge") is True
        and meta.get("strict_only") is True
        and meta.get("judge_repeats") == evaluation["judge_repeats"]
        and meta.get("fail_on_judge_error") is True
        and status.get("available") is True
        and status.get("selected_model") == evaluation["judge_model"]
        and (meta.get("strict_judge_sources") or {}).get("api_median") == expected
    )


def preserve_incomplete(path: Path) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.exists():
        target = path.with_name(f"{path.name}.incomplete_{stamp}")
        path.replace(target)
    for suffix in [".meta.json", ".score_meta.json", ".failed_score_meta.json"]:
        sibling = path.with_suffix(suffix)
        if sibling.exists():
            sibling.replace(sibling.with_name(f"{sibling.name}.incomplete_{stamp}"))


def execute(command: list[str], log_path: Path, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        safe_env = {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]} if env and "CUDA_VISIBLE_DEVICES" in env else {}
        print(json.dumps({"command": command, "log": str(log_path), "env": safe_env}, ensure_ascii=False))
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"started_at": utc_now(), "command": command}, ensure_ascii=False) + "\n")
        log.flush()
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def raw_command(
    python: str,
    checkpoint: dict[str, Any],
    benchmark: dict[str, Any],
    output: Path,
    run_id: str,
    mode: str,
    evaluation: dict[str, Any],
) -> list[str]:
    common = [
        "--checkpoint",
        checkpoint["checkpoint"],
        "--model-label",
        checkpoint["id"],
        "--instances",
        benchmark["instances"],
        "--output",
        str(output),
        "--run-id",
        run_id,
        "--device",
        "cuda:0",
        "--device-map",
        "single",
        "--dtype",
        evaluation["dtype"],
        "--max-new-tokens",
        str(evaluation["max_new_tokens"]),
        "--query-style",
        evaluation["query_style"],
        "--fail-on-audit-issue",
    ]
    if mode == "smoke":
        common.extend(["--limit", "1"])
    if benchmark["lane"] == "memqa":
        return [python, "-m", "eval.benchmarks.memqa.scripts.run_metis_memqa", *common]
    return [
        python,
        "-m",
        "eval.benchmarks.memops.scripts.run_memop_memory_baseline",
        "--method",
        "metis",
        *common,
        "--oom-policy",
        evaluation["memop_oom_policy"],
    ]


def run_checkpoint_raw(
    checkpoint: dict[str, Any],
    benchmarks: list[dict[str, Any]],
    run_dir: Path,
    run_id: str,
    mode: str,
    gpu: str,
    python: str,
    evaluation: dict[str, Any],
    dry_run: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        output = raw_path(run_dir, checkpoint["id"], benchmark["id"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if raw_complete(output, benchmark, mode) and raw_meta_complete(output, checkpoint):
            events.append({"checkpoint": checkpoint["id"], "benchmark": benchmark["id"], "status": "skip_complete"})
            continue
        preserve_incomplete(output)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = raw_command(
            python,
            checkpoint,
            benchmark,
            output,
            f"{run_id}_{checkpoint['id']}_{benchmark['id']}",
            mode,
            evaluation,
        )
        log = run_dir / "logs" / checkpoint["id"] / f"{benchmark['id']}.raw.log"
        execute(command, log, env=env, dry_run=dry_run)
        if not dry_run and not (
            raw_complete(output, benchmark, mode) and raw_meta_complete(output, checkpoint)
        ):
            raise RuntimeError(f"Raw output or delta load metadata failed completeness audit: {output}")
        events.append({"checkpoint": checkpoint["id"], "benchmark": benchmark["id"], "status": "dry_run" if dry_run else "complete"})
    return events


def run_raw_stage(args: argparse.Namespace, matrix: dict[str, Any]) -> dict[str, Any]:
    if args.mode == "full" and args.axis == "all":
        raise ValueError("Full runs must use --axis data or --axis structure so result roots remain separated")
    selected = selected_checkpoints(
        matrix, args.axis, args.checkpoint, args.include_pending, args.mode, args.rerun_reference
    )
    if not selected:
        raise ValueError("No checkpoints selected")
    benchmarks = selected_benchmarks(matrix, args.include_appendix_benchmarks)
    if args.benchmark:
        available = {item["id"] for item in benchmarks}
        unknown = sorted(set(args.benchmark) - available)
        if unknown:
            raise ValueError(f"Unknown or excluded benchmark ids: {unknown}")
        benchmarks = [item for item in benchmarks if item["id"] in args.benchmark]
    audit = run_static_audit(
        matrix, args.matrix, args.run_dir / "audit" / "preflight.json"
    )
    selected_ids = {item["id"] for item in selected}
    failed = [
        item["id"]
        for item in audit["checkpoints"]
        if item["id"] in selected_ids and not item["ok"]
    ]
    if failed:
        raise RuntimeError(f"Selected checkpoint audit failed: {failed}")

    gpu_map = parse_gpu_map(args.gpu_map, args.gpu)
    cell_gpu: dict[str, str] = {}
    for checkpoint in selected:
        for benchmark in benchmarks:
            cell_id = f"{checkpoint['id']}/{benchmark['id']}"
            gpu = gpu_map.get(cell_id, gpu_map.get(checkpoint["id"], gpu_map.get("*")))
            if gpu is None:
                raise ValueError(f"No GPU assigned for {cell_id}")
            cell_gpu[cell_id] = gpu

    valid_gpu_keys = set(cell_gpu) | {item["id"] for item in selected} | {"*"}
    unknown_gpu_keys = sorted(set(gpu_map) - valid_gpu_keys)
    if unknown_gpu_keys:
        raise ValueError(f"Unknown --gpu-map keys: {unknown_gpu_keys}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_id": args.run_dir.name,
        "created_at": utc_now(),
        "branch": matrix["branch_id"],
        "ownership": matrix["ownership"],
        "axis": args.axis,
        "mode": args.mode,
        "matrix": str(args.matrix),
        "matrix_sha256": sha256(args.matrix),
        "git": git_snapshot(),
        "metis_dev_git": git_snapshot(METIS_DEV_ROOT),
        "python": args.python,
        "checkpoints": selected,
        "benchmarks": benchmarks,
        "main_text_benchmark_ids": matrix.get("main_text_benchmark_ids"),
        "appendix_benchmark_ids": matrix.get("appendix_benchmark_ids"),
        "include_appendix_benchmarks": args.include_appendix_benchmarks,
        "evaluation": matrix["evaluation"],
        "gpu_map": cell_gpu,
        "rerun_reference": args.rerun_reference,
        "result_status": "pending_review_and_promotion",
    }
    write_json(args.run_dir / "run_config.json", run_config)

    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for checkpoint in selected:
        for benchmark in benchmarks:
            cell_id = f"{checkpoint['id']}/{benchmark['id']}"
            groups[cell_gpu[cell_id]].append((checkpoint, benchmark))

    events: list[dict[str, Any]] = []

    def run_gpu_group(
        gpu: str, cells: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        group_events: list[dict[str, Any]] = []
        for checkpoint, benchmark in cells:
            group_events.extend(
                run_checkpoint_raw(
                    checkpoint,
                    [benchmark],
                    args.run_dir,
                    args.run_dir.name,
                    args.mode,
                    gpu,
                    args.python,
                    matrix["evaluation"],
                    args.dry_run,
                )
            )
        return group_events

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(run_gpu_group, gpu, cells) for gpu, cells in groups.items()]
        for future in concurrent.futures.as_completed(futures):
            events.extend(future.result())
    write_json(args.run_dir / "audit" / "raw_stage.json", {"created_at": utc_now(), "events": events})
    return {"events": events, "run_dir": str(args.run_dir)}


def run_score_stage(args: argparse.Namespace, matrix: dict[str, Any]) -> dict[str, Any]:
    if args.judge_concurrency < 1:
        raise ValueError("--judge-concurrency is required and must be >= 1 for scoring")
    config_path = args.run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    run_config = load_json(config_path)
    mode = run_config["mode"]
    checkpoints = run_config["checkpoints"]
    benchmarks = run_config["benchmarks"]
    evaluation = run_config["evaluation"]
    events = []
    score_env = load_dotenv(os.environ.copy())
    args.judge_lock.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "judge_lock_wait", "path": str(args.judge_lock)}, ensure_ascii=False), flush=True)
    with args.judge_lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        write_json(
            args.run_dir / "audit" / "judge_lock.json",
            {"acquired_at": utc_now(), "path": str(args.judge_lock), "judge_concurrency": args.judge_concurrency},
        )
        print(json.dumps({"event": "judge_lock_acquired", "path": str(args.judge_lock)}, ensure_ascii=False), flush=True)
        for checkpoint in checkpoints:
            for benchmark in benchmarks:
                raw = raw_path(args.run_dir, checkpoint["id"], benchmark["id"])
                if not (raw_complete(raw, benchmark, mode) and raw_meta_complete(raw, checkpoint)):
                    raise RuntimeError(f"Raw file or delta load metadata is not complete: {raw}")
                output = scored_path(args.run_dir, checkpoint["id"], benchmark["id"])
                output.parent.mkdir(parents=True, exist_ok=True)
                if scored_complete(output, benchmark, mode, evaluation):
                    events.append({"checkpoint": checkpoint["id"], "benchmark": benchmark["id"], "status": "skip_complete"})
                    continue
                preserve_incomplete(output)
                command = [
                    args.python,
                    "-m",
                    "eval.benchmarks.memqa.scripts.score_memqa",
                    "--instances",
                    benchmark["instances"],
                    "--input",
                    str(raw),
                    "--output",
                    str(output),
                    "--judge-base-url",
                    args.judge_base_url,
                    "--judge-model",
                    evaluation["judge_model"],
                    "--api-key-env",
                    args.api_key_env,
                    "--judge-repeats",
                    str(evaluation["judge_repeats"]),
                    "--strict-judge",
                    "--strict-only",
                    "--fail-on-judge-error",
                    "--concurrency",
                    str(args.judge_concurrency),
                    "--judge-max-attempts",
                    str(args.judge_max_attempts),
                    "--judge-retry-sleep",
                    str(args.judge_retry_sleep),
                    "--judge-retry-backoff",
                    str(args.judge_retry_backoff),
                ]
                log = args.run_dir / "logs" / checkpoint["id"] / f"{benchmark['id']}.score.log"
                execute(command, log, env=score_env, dry_run=args.dry_run)
                if not args.dry_run and not scored_complete(output, benchmark, mode, evaluation):
                    raise RuntimeError(f"Scored output failed completeness audit: {output}")
                events.append({"checkpoint": checkpoint["id"], "benchmark": benchmark["id"], "status": "dry_run" if args.dry_run else "complete"})
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        print(json.dumps({"event": "judge_lock_released", "path": str(args.judge_lock)}, ensure_ascii=False), flush=True)
    write_json(args.run_dir / "audit" / "score_stage.json", {"created_at": utc_now(), "events": events})
    return {"events": events, "run_dir": str(args.run_dir)}


def result_audit(args: argparse.Namespace, matrix: dict[str, Any]) -> dict[str, Any]:
    run_config = load_json(args.run_dir / "run_config.json")
    mode = run_config["mode"]
    evaluation = run_config["evaluation"]
    cells = []
    for checkpoint in run_config["checkpoints"]:
        for benchmark in run_config["benchmarks"]:
            raw = raw_path(args.run_dir, checkpoint["id"], benchmark["id"])
            scored = scored_path(args.run_dir, checkpoint["id"], benchmark["id"])
            raw_scan = jsonl_scan(raw) if raw.is_file() else None
            scored_scan = jsonl_scan(scored) if scored.is_file() else None
            cell = {
                "checkpoint": checkpoint["id"],
                "benchmark": benchmark["id"],
                "raw": str(raw),
                "scored": str(scored),
                "raw_complete": raw_complete(raw, benchmark, mode),
                "raw_meta_complete": raw_meta_complete(raw, checkpoint) if raw.is_file() else False,
                "scored_complete": scored_complete(scored, benchmark, mode, evaluation),
                "raw_scan": raw_scan,
                "scored_scan": scored_scan,
            }
            cell["ok"] = cell["raw_complete"] and cell["raw_meta_complete"] and cell["scored_complete"]
            cells.append(cell)
    report = {"created_at": utc_now(), "cells": cells, "ok": all(item["ok"] for item in cells)}
    write_json(args.run_dir / "audit" / "result_audit.json", report)
    return report


def axis_summary(args: argparse.Namespace, matrix: dict[str, Any]) -> dict[str, Any]:
    run_config = load_json(args.run_dir / "run_config.json")
    axis = run_config["axis"]
    if axis not in {"data", "structure"}:
        raise ValueError(
            "Summary requires a run created with --axis data or structure"
        )
    reference_id = matrix["reference_checkpoint_id"]
    reference_scores = matrix["formal_reference_scores"]
    checkpoint_map, benchmark_map = matrix_maps(matrix)
    main_text_ids = set(matrix.get("main_text_benchmark_ids") or benchmark_map)
    run_benchmark_ids = [item["id"] for item in run_config["benchmarks"]]
    checkpoint_ids = [item["id"] for item in run_config["checkpoints"]]
    reuse_map = parse_reuse_scored(args.reuse_scored)
    unknown_reuse = sorted(set(reuse_map) - set(checkpoint_map))
    if unknown_reuse:
        raise ValueError(f"Unknown --reuse-scored checkpoint ids: {unknown_reuse}")
    overlap = sorted(set(reuse_map) & set(checkpoint_ids))
    if overlap:
        raise ValueError(f"Reused checkpoints already belong to this run: {overlap}")

    for checkpoint_id, source_run_dir in reuse_map.items():
        source_audit_path = source_run_dir / "audit" / "result_audit.json"
        source_config_path = source_run_dir / "run_config.json"
        if not source_audit_path.is_file() or not load_json(source_audit_path).get("ok"):
            raise RuntimeError(f"Reused run lacks a passing result audit: {source_run_dir}")
        if not source_config_path.is_file():
            raise RuntimeError(f"Reused run lacks run_config.json: {source_run_dir}")
        source_config = load_json(source_config_path)
        if source_config.get("mode") != "full":
            raise RuntimeError(f"Reused run is not a full run: {source_run_dir}")
        source_checkpoints = {item["id"]: item for item in source_config["checkpoints"]}
        if checkpoint_id not in source_checkpoints:
            raise RuntimeError(
                f"Reused run does not contain checkpoint {checkpoint_id}: {source_run_dir}"
            )
        source_checkpoint = source_checkpoints[checkpoint_id]
        current_checkpoint = checkpoint_map[checkpoint_id]
        for key in ["checkpoint", "delta_sha256", "config_sha256"]:
            if source_checkpoint.get(key) != current_checkpoint.get(key):
                raise RuntimeError(
                    f"Reused checkpoint provenance mismatch for {checkpoint_id}: {key}"
                )
        source_benchmarks = {item["id"]: item for item in source_config["benchmarks"]}
        for benchmark_id in run_benchmark_ids:
            if benchmark_id not in source_benchmarks:
                raise RuntimeError(
                    f"Reused run lacks benchmark {benchmark_id}: {source_run_dir}"
                )
            source_benchmark = source_benchmarks[benchmark_id]
            current_benchmark = benchmark_map[benchmark_id]
            for key in ["rows", "sha256", "first_instance_id", "last_instance_id"]:
                if source_benchmark.get(key) != current_benchmark.get(key):
                    raise RuntimeError(
                        f"Reused benchmark provenance mismatch for {benchmark_id}: {key}"
                    )
            source_scored = scored_path(source_run_dir, checkpoint_id, benchmark_id)
            if not scored_complete(
                source_scored,
                source_benchmark,
                source_config["mode"],
                source_config["evaluation"],
            ):
                raise RuntimeError(f"Reused scored cell is incomplete: {source_scored}")
        checkpoint_ids.append(checkpoint_id)
    if reference_id not in checkpoint_ids:
        checkpoint_ids.append(reference_id)
    rows = []
    for checkpoint_id in checkpoint_ids:
        checkpoint = checkpoint_map[checkpoint_id]
        for benchmark_id in run_benchmark_ids:
            benchmark = benchmark_map[benchmark_id]
            scored = scored_path(args.run_dir, checkpoint_id, benchmark_id)
            if scored.is_file():
                strict = jsonl_scan(scored)["strict_mean"]
                provenance = str(scored)
            elif checkpoint_id in reuse_map:
                reused_scored = scored_path(reuse_map[checkpoint_id], checkpoint_id, benchmark_id)
                strict = jsonl_scan(reused_scored)["strict_mean"]
                provenance = str(reused_scored)
            elif checkpoint_id == reference_id:
                strict = float(reference_scores[benchmark_id])
                provenance = "external/formal-reference-results-20260711"
            else:
                continue
            rows.append(
                {
                    "axis": axis,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_label": checkpoint["label"],
                    "benchmark": benchmark_id,
                    "lane": benchmark["lane"],
                    "report_scope": "main_text" if benchmark_id in main_text_ids else "appendix",
                    "strict": strict,
                    "reference_strict": float(reference_scores[benchmark_id]),
                    "absolute_delta_vs_v24_gdn": strict - float(reference_scores[benchmark_id]),
                    "comparability": checkpoint["comparability"],
                    "provenance": provenance,
                }
            )

    aggregates = []
    for checkpoint_id in checkpoint_ids:
        checkpoint_rows = [row for row in rows if row["checkpoint_id"] == checkpoint_id]
        if not checkpoint_rows:
            continue
        main_rows = [row for row in checkpoint_rows if row["report_scope"] == "main_text"]
        for name, selected in [
            ("memqa_main_macro", [row for row in main_rows if row["lane"] == "memqa"]),
            ("memop_main_macro", [row for row in main_rows if row["lane"] == "memop"]),
            ("four_benchmark_main_macro", main_rows),
        ]:
            if selected:
                aggregates.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "aggregate": name,
                        "strict": sum(row["strict"] for row in selected) / len(selected),
                        "reference_strict": sum(row["reference_strict"] for row in selected) / len(selected),
                        "absolute_delta_vs_v24_gdn": sum(row["absolute_delta_vs_v24_gdn"] for row in selected) / len(selected),
                    }
                )
    summary = {
        "created_at": utc_now(),
        "axis": axis,
        "reporting_matrix_sha256": sha256(args.matrix),
        "run_config_matrix_sha256": run_config.get("matrix_sha256"),
        "main_text_benchmark_ids": list(matrix.get("main_text_benchmark_ids") or benchmark_map),
        "appendix_benchmark_ids": list(matrix.get("appendix_benchmark_ids") or []),
        "reuse_scored": {key: str(value) for key, value in reuse_map.items()},
        "rows": rows,
        "aggregates": aggregates,
    }
    output_dir = args.run_dir / "summary"
    write_json(output_dir / f"{axis}_ablation_summary.json", summary)
    title = f"Metis {axis.title()} Ablation Summary"
    lines = [
        f"# {title}",
        "",
        "Evaluation result pending review and promotion; not a formal/current table.",
        "",
        "Main-text aggregates exclude `metisops_v23_full_segments`; that benchmark is appendix-only.",
        "",
        "| checkpoint | benchmark | strict | v2.4 GDN | delta | comparability |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in [item for item in rows if item["report_scope"] == "main_text"]:
        lines.append(
            f"| `{row['checkpoint_id']}` | `{row['benchmark']}` | {row['strict']:.6f} | "
            f"{row['reference_strict']:.6f} | {row['absolute_delta_vs_v24_gdn']:+.6f} | {row['comparability']} |"
        )
    appendix_rows = [item for item in rows if item["report_scope"] == "appendix"]
    if appendix_rows:
        lines.extend([
            "",
            "## Appendix-only benchmarks",
            "",
            "| checkpoint | benchmark | strict | v2.4 GDN | delta | comparability |",
            "|---|---|---:|---:|---:|---|",
        ])
        for row in appendix_rows:
            lines.append(
                f"| `{row['checkpoint_id']}` | `{row['benchmark']}` | {row['strict']:.6f} | "
                f"{row['reference_strict']:.6f} | {row['absolute_delta_vs_v24_gdn']:+.6f} | {row['comparability']} |"
            )
    lines.extend(["", "## Main-text macro averages", "", "| checkpoint | aggregate | strict | v2.4 GDN | delta |", "|---|---|---:|---:|---:|"])
    for row in aggregates:
        lines.append(
            f"| `{row['checkpoint_id']}` | `{row['aggregate']}` | {row['strict']:.6f} | "
            f"{row['reference_strict']:.6f} | {row['absolute_delta_vs_v24_gdn']:+.6f} |"
        )
    (output_dir / f"{axis}_ablation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["audit", "raw", "score", "result-audit", "summary"])
    parser.add_argument(
        "--axis",
        default="all",
        choices=["data", "structure", "all"],
    )
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSET_REGISTRY)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--gpu-map", action="append", default=[])
    parser.add_argument("--gpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--include-appendix-benchmarks", action="store_true")
    parser.add_argument("--rerun-reference", action="store_true")
    parser.add_argument("--judge-concurrency", type=int, default=0)
    parser.add_argument("--judge-base-url", default="https://api.openai.com")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-lock", type=Path, default=Path("/tmp/metis_eval_judge32.lock"))
    parser.add_argument("--judge-max-attempts", type=int, default=18)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)
    parser.add_argument("--judge-retry-backoff", type=float, default=1.5)
    parser.add_argument(
        "--reuse-scored",
        action="append",
        default=[],
        help="For summary only: CHECKPOINT_ID=RUN_DIR with a passing full result audit",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.matrix = args.matrix.resolve()
    matrix = load_json(args.matrix)
    assets = load_assets(args.assets)
    for checkpoint in matrix["checkpoints"]:
        checkpoint["checkpoint"] = resolve_asset(checkpoint["id"], assets)
    if args.stage == "audit":
        output = args.run_dir / "audit" / "preflight.json" if args.run_dir else None
        result = run_static_audit(matrix, args.matrix, output)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required for this stage")
        args.run_dir = args.run_dir.resolve()
        if args.stage == "raw":
            result = run_raw_stage(args, matrix)
        elif args.stage == "score":
            result = run_score_stage(args, matrix)
        elif args.stage == "result-audit":
            result = result_audit(args, matrix)
        else:
            result = axis_summary(args, matrix)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.stage in {"audit", "result-audit"} and not result.get("ready_ok", result.get("ok", True)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
