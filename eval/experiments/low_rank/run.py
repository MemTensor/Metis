#!/usr/bin/env python3
"""Run and score the Metis LocalMemory low-rank full sweep.

Designed for durable Metis evaluation runs. Raw inference uses one Metis process per GPU.
Scoring is serialized by file so total judge concurrency stays bounded.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
DEFAULT_CKPT: Path | None = None
CKPT: Path | None = DEFAULT_CKPT
DEFAULT_MODEL_LABEL_PREFIX = "metis_lowrank"
MODEL_LABEL_PREFIX = DEFAULT_MODEL_LABEL_PREFIX

DEFAULT_RANKS: list[int | str] = [1, 4, 16, 64, 128, 256, "full"]
RANKS = list(DEFAULT_RANKS)


@dataclass(frozen=True)
class Benchmark:
    name: str
    lane: str
    instances: Path
    expected_rows: int


BENCHMARKS = [
    Benchmark("locomo_tps16", "memqa", Path("memqa/locomo_evidence_sessions_tps16_20260626.jsonl"), 1527),
    Benchmark("nextmem_stm", "memqa", Path("memqa/nextmem_stm_official_task2_20260622.jsonl"), 1697),
    Benchmark("metis_test", "memop", Path("memops/metis_test_v2_4_memoryops_test_nomixed_20260707.jsonl"), 852),
    Benchmark("metisops_v23_gold_turns", "memop", Path("memops/metisops_v23_gold_turns_20260626.jsonl"), 531),
]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_status(run_dir: Path, message: str) -> None:
    status = run_dir / "status.txt"
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")
        handle.flush()


def rank_label(rank: int | str) -> str:
    return "rankfull" if rank == "full" else f"rank{int(rank):04d}"


def model_label(rank: int | str) -> str:
    return f"{MODEL_LABEL_PREFIX}_{rank_label(rank)}"


def parse_rank_grid(value: str) -> list[int | str]:
    ranks: list[int | str] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item in {"full", "none", "no-projection", "no_projection"}:
            ranks.append("full")
            continue
        rank = int(item)
        if rank <= 0:
            raise ValueError(f"numeric rank must be positive, got {rank}")
        ranks.append(rank)
    if not ranks:
        raise ValueError("rank grid is empty")
    return ranks


def raw_path(run_dir: Path, bench: Benchmark, rank: int | str) -> Path:
    return run_dir / "raw" / bench.name / f"{model_label(rank)}.raw.jsonl"


def scored_path(run_dir: Path, bench: Benchmark, rank: int | str) -> Path:
    return run_dir / "scored_strict" / bench.name / f"{model_label(rank)}.scored.jsonl"


def log_path(run_dir: Path, bench: Benchmark, rank: int | str, suffix: str) -> Path:
    return run_dir / "logs" / bench.name / f"{model_label(rank)}.{suffix}.log"


def raw_complete(path: Path, expected_rows: int) -> bool:
    return line_count(path) == expected_rows


def scored_complete(path: Path, expected_rows: int) -> bool:
    return line_count(path) == expected_rows


def raw_command(args: argparse.Namespace, bench: Benchmark, rank: int | str, output: Path, run_id: str) -> list[str]:
    common = [
        "--model-label",
        model_label(rank),
        "--instances",
        str(bench.instances),
        "--output",
        str(output),
        "--run-id",
        run_id,
        "--device",
        args.device,
        "--device-map",
        args.device_map,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--query-style",
        args.query_style,
        "--fail-on-audit-issue",
        "--limit",
        str(args.limit),
    ]
    if rank != "full":
        common.extend(
            [
                "--metis-low-rank-local-memory",
                "--metis-low-rank-rank",
                str(rank),
                "--metis-low-rank-policy",
                "after_each_commit",
                "--metis-low-rank-target",
                "state",
            ]
        )
    if bench.lane == "memqa":
        return [
            str(PYTHON),
            "-m",
            "eval.benchmarks.memqa.scripts.run_metis_memqa",
            "--checkpoint",
            str(args.checkpoint),
            *common,
        ]
    return [
        str(PYTHON),
        "-m",
        "eval.benchmarks.memops.scripts.run_memop_memory_baseline",
        "--method",
        "metis",
        "--checkpoint",
        str(args.checkpoint),
        "--oom-policy",
        "fail",
        *common,
    ]


def score_command(args: argparse.Namespace, bench: Benchmark, raw: Path, scored: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "eval.benchmarks.memqa.scripts.score_memqa",
        "--instances",
        str(bench.instances),
        "--input",
        str(raw),
        "--output",
        str(scored),
        "--judge-base-url",
        args.judge_base_url,
        "--judge-model",
        args.judge_model,
        "--api-key-env",
        args.judge_api_key_env,
        "--judge-repeats",
        str(args.judge_repeats),
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
        "--progress-every",
        str(args.progress_every),
    ]


def run_subprocess(cmd: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] CMD {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
        return proc.wait()


def validate_inputs(*, limited: bool = False) -> None:
    for bench in BENCHMARKS:
        actual = line_count(bench.instances)
        if (limited and actual < bench.expected_rows) or (not limited and actual != bench.expected_rows):
            raise RuntimeError(f"{bench.name} expected {bench.expected_rows} rows, found {actual}: {bench.instances}")
    if CKPT is None or not CKPT.exists():
        raise RuntimeError(f"checkpoint missing: {CKPT}")
    if not PYTHON.exists():
        raise RuntimeError(f"python missing: {PYTHON}")


def write_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank in RANKS:
        for bench in BENCHMARKS:
            rows.append(
                {
                    "rank": rank,
                    "rank_label": rank_label(rank),
                    "benchmark": bench.name,
                    "lane": bench.lane,
                    "instances": str(bench.instances),
                    "expected_rows": bench.expected_rows,
                    "raw": str(raw_path(run_dir, bench, rank)),
                    "scored_strict": str(scored_path(run_dir, bench, rank)),
                }
            )
    write_json(
        run_dir / "run_config.json",
        {
            "run_id": args.run_id,
            "run_dir": str(run_dir),
            "created_at": utc_now(),
            "checkpoint": str(args.checkpoint),
            "model_label_prefix": args.model_label_prefix,
            "ranks": RANKS,
            "policy": "full is no low-rank; numeric ranks use after_each_commit on LocalMemory state",
            "benchmarks": [bench.__dict__ | {"instances": str(bench.instances)} for bench in BENCHMARKS],
            "runtime": {
                "device": args.device,
                "device_map": args.device_map,
                "dtype": args.dtype,
                "query_style": args.query_style,
                "max_new_tokens": args.max_new_tokens,
                "gpu_ids": args.gpu_ids,
                "raw_workers": args.raw_workers,
                "limit": args.limit,
            },
            "scoring": {
                "judge_model": args.judge_model,
                "judge_repeats": args.judge_repeats,
                "strict_only": True,
                "judge_concurrency": args.judge_concurrency,
                "judge_base_url": args.judge_base_url,
                "judge_api_key_env": args.judge_api_key_env,
            },
        },
    )
    manifest = run_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def raw_tasks(args: argparse.Namespace, run_dir: Path) -> list[tuple[Benchmark, int | str]]:
    tasks = []
    for rank in RANKS:
        for bench in BENCHMARKS:
            output = raw_path(run_dir, bench, rank)
            if args.resume and raw_complete(output, bench.expected_rows):
                append_status(run_dir, f"SKIP raw complete {bench.name} {rank_label(rank)}")
                continue
            tasks.append((bench, rank))
    return tasks


def run_raw(args: argparse.Namespace, run_dir: Path) -> None:
    tasks = raw_tasks(args, run_dir)
    append_status(run_dir, f"RAW_START pending={len(tasks)}")
    if not tasks:
        append_status(run_dir, "RAW_DONE nothing_to_do")
        return
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    workers = min(args.raw_workers or len(gpu_ids), len(gpu_ids), len(tasks))
    task_queue: queue.Queue[tuple[Benchmark, int | str]] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    def worker(worker_idx: int) -> list[dict[str, Any]]:
        gpu = gpu_ids[worker_idx % len(gpu_ids)]
        results = []
        while True:
            try:
                bench, rank = task_queue.get_nowait()
            except queue.Empty:
                break
            out = raw_path(run_dir, bench, rank)
            log = log_path(run_dir, bench, rank, "raw")
            run_id = f"{args.run_id}_{bench.name}_{rank_label(rank)}"
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            append_status(run_dir, f"RAW_TASK_START gpu={gpu} bench={bench.name} rank={rank_label(rank)}")
            started = time.time()
            rc = run_subprocess(raw_command(args, bench, rank, out, run_id), log, env)
            rows = line_count(out)
            result = {
                "bench": bench.name,
                "rank": rank,
                "gpu": gpu,
                "return_code": rc,
                "rows": rows,
                "expected_rows": bench.expected_rows,
                "elapsed_sec": round(time.time() - started, 3),
                "output": str(out),
                "log": str(log),
            }
            results.append(result)
            append_status(run_dir, f"RAW_TASK_DONE gpu={gpu} bench={bench.name} rank={rank_label(rank)} rc={rc} rows={rows}/{bench.expected_rows}")
            if rc != 0 or rows != bench.expected_rows:
                raise RuntimeError(f"raw failed: {result}")
            task_queue.task_done()
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, idx) for idx in range(workers)]
        all_results = []
        for future in as_completed(futures):
            all_results.extend(future.result())
    write_json(run_dir / "summary" / "raw_results.json", all_results)
    append_status(run_dir, "RAW_DONE")


def score_tasks(args: argparse.Namespace, run_dir: Path) -> list[tuple[Benchmark, int | str]]:
    tasks = []
    for rank in RANKS:
        for bench in BENCHMARKS:
            raw = raw_path(run_dir, bench, rank)
            scored = scored_path(run_dir, bench, rank)
            if not raw_complete(raw, bench.expected_rows):
                continue
            if args.resume and scored_complete(scored, bench.expected_rows):
                append_status(run_dir, f"SKIP score complete {bench.name} {rank_label(rank)}")
                continue
            tasks.append((bench, rank))
    return tasks


def completion_counts(run_dir: Path) -> dict[str, int]:
    total = len(RANKS) * len(BENCHMARKS)
    raw_done = 0
    scored_done = 0
    for rank in RANKS:
        for bench in BENCHMARKS:
            if raw_complete(raw_path(run_dir, bench, rank), bench.expected_rows):
                raw_done += 1
            if scored_complete(scored_path(run_dir, bench, rank), bench.expected_rows):
                scored_done += 1
    return {"total": total, "raw_done": raw_done, "scored_done": scored_done}


def run_score(args: argparse.Namespace, run_dir: Path) -> None:
    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise RuntimeError(f"Judge API key env is not set: {args.judge_api_key_env}")
    tasks = score_tasks(args, run_dir)
    append_status(run_dir, f"SCORE_START pending={len(tasks)} concurrency={args.judge_concurrency}")
    results = []
    for bench, rank in tasks:
        raw = raw_path(run_dir, bench, rank)
        scored = scored_path(run_dir, bench, rank)
        log = log_path(run_dir, bench, rank, "score_strict")
        scored.parent.mkdir(parents=True, exist_ok=True)
        append_status(run_dir, f"SCORE_TASK_START bench={bench.name} rank={rank_label(rank)}")
        started = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        rc = run_subprocess(score_command(args, bench, raw, scored), log, env)
        rows = line_count(scored)
        result = {
            "bench": bench.name,
            "rank": rank,
            "return_code": rc,
            "rows": rows,
            "expected_rows": bench.expected_rows,
            "elapsed_sec": round(time.time() - started, 3),
            "output": str(scored),
            "log": str(log),
        }
        results.append(result)
        append_status(run_dir, f"SCORE_TASK_DONE bench={bench.name} rank={rank_label(rank)} rc={rc} rows={rows}/{bench.expected_rows}")
        if rc != 0 or rows != bench.expected_rows:
            raise RuntimeError(f"score failed: {result}")
    write_json(run_dir / "summary" / "score_results.json", results)
    append_status(run_dir, "SCORE_DONE")


def run_score_watch(args: argparse.Namespace, run_dir: Path) -> None:
    append_status(run_dir, f"SCORE_WATCH_START poll_seconds={args.score_poll_seconds}")
    while True:
        counts = completion_counts(run_dir)
        append_status(
            run_dir,
            f"SCORE_WATCH_TICK raw={counts['raw_done']}/{counts['total']} scored={counts['scored_done']}/{counts['total']}",
        )
        if counts["scored_done"] >= counts["total"]:
            summarize(args, run_dir)
            append_status(run_dir, "SCORE_WATCH_DONE all_scored")
            return
        run_score(args, run_dir)
        summarize(args, run_dir)
        counts = completion_counts(run_dir)
        if counts["scored_done"] >= counts["total"]:
            append_status(run_dir, "SCORE_WATCH_DONE all_scored")
            return
        time.sleep(max(60, args.score_poll_seconds))


def summarize(args: argparse.Namespace, run_dir: Path) -> None:
    scored_files = [
        scored_path(run_dir, bench, rank)
        for rank in RANKS
        for bench in BENCHMARKS
        if scored_complete(scored_path(run_dir, bench, rank), bench.expected_rows)
    ]
    rows = []
    for path in scored_files:
        with path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            continue
        strict_scores = [float(row.get("score", {}).get("llm_judge_strict_score") or 0.0) for row in records]
        audit_issues = sum(len(row.get("audit_issues") or []) for row in records)
        runtime_errors = sum(1 for row in records if str(row.get("runtime_status") or "ok") != "ok")
        first = records[0]
        rows.append(
            {
                "file": str(path),
                "benchmark": first.get("dataset") or path.parent.name,
                "path_benchmark": path.parent.name,
                "model_label": first.get("model_label"),
                "rank_label": str(first.get("model_label", "")).rsplit("_", 1)[-1],
                "records": len(records),
                "strict_mean": sum(strict_scores) / len(strict_scores),
                "strict_pass_rate": sum(1 for score in strict_scores if score >= 0.75) / len(strict_scores),
                "audit_issues": audit_issues,
                "runtime_errors": runtime_errors,
            }
        )
    write_json(run_dir / "summary" / "strict_summary.json", rows)
    md = [
        "# Metis LocalMemory Low-Rank Strict Summary",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "| benchmark | model_label | records | strict_mean | strict_pass_rate | audit_issues | runtime_errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["path_benchmark"], item["model_label"])):
        md.append(
            f"| `{row['path_benchmark']}` | `{row['model_label']}` | {row['records']} | "
            f"{row['strict_mean']:.4f} | {row['strict_pass_rate']:.4f} | "
            f"{row['audit_issues']} | {row['runtime_errors']} |"
        )
    (run_dir / "summary" / "strict_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    append_status(run_dir, f"SUMMARY_DONE scored_files={len(scored_files)}")


def main() -> None:
    global CKPT, MODEL_LABEL_PREFIX, RANKS, BENCHMARKS, PYTHON
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-label-prefix", default=DEFAULT_MODEL_LABEL_PREFIX)
    parser.add_argument("--ranks", default=",".join(str(item) for item in DEFAULT_RANKS))
    parser.add_argument("--benchmark", action="append", choices=[item.name for item in BENCHMARKS], default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stage", choices=["raw", "score", "score-watch", "summary", "all"], default="all")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--raw-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="single", choices=["single", "paired_layers", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--query-style", default="memory_direct", choices=["default", "memory_direct", "minimal"])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", "https://api.openai.com"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--judge-api-key-env", default=os.environ.get("JUDGE_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--judge-max-attempts", type=int, default=18)
    parser.add_argument("--judge-retry-sleep", type=float, default=2.0)
    parser.add_argument("--judge-retry-backoff", type=float, default=1.5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--score-poll-seconds", type=int, default=900)
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.data_dir is None:
        parser.error("--data-dir is required")
    if args.checkpoint is None and not args.dry_run:
        parser.error("--checkpoint is required unless --dry-run is used")
    args.checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else Path("CHECKPOINT_REQUIRED")
    CKPT = args.checkpoint
    PYTHON = args.python.expanduser().resolve()
    MODEL_LABEL_PREFIX = args.model_label_prefix
    RANKS = parse_rank_grid(args.ranks)
    data_dir = args.data_dir.expanduser().resolve()
    BENCHMARKS = [
        Benchmark(
            item.name,
            item.lane,
            data_dir / item.instances,
            min(args.limit, item.expected_rows) if args.limit else item.expected_rows,
        )
        for item in BENCHMARKS
        if not args.benchmark or item.name in args.benchmark
    ]

    run_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if args.dry_run:
        write_manifest(run_dir, args)
        write_json(
            run_dir / "dry_run.json",
            {
                "experiment": "figure_4_table_10_low_rank",
                "ranks": RANKS,
                "benchmarks": [item.name for item in BENCHMARKS],
            },
        )
        return
    validate_inputs(limited=bool(args.limit))
    write_manifest(run_dir, args)
    append_status(run_dir, f"START stage={args.stage}")

    if args.stage in {"raw", "all"}:
        run_raw(args, run_dir)
    if args.stage in {"score", "all"}:
        run_score(args, run_dir)
    if args.stage == "score-watch":
        run_score_watch(args, run_dir)
    if args.stage in {"summary", "all"}:
        summarize(args, run_dir)
    append_status(run_dir, f"DONE stage={args.stage}")


if __name__ == "__main__":
    main()
