#!/usr/bin/env python3
"""Score one frozen MemQA-OOD cell with official task metrics.

Semantic rows use only the authorized MemTensor gpt-4.1-mini judge, three
strict repeats, with the benchmark's official rubric. The output is resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from eval.common.judge import JudgeClient
from eval.common.paths import REPO_ROOT


ATM_JUDGE_PROMPT = ""
ATM_VENDOR_MANIFEST = REPO_ROOT / "eval/third_party/atm_bench/MANIFEST.json"
atm_number_core: Any = None
atm_list_core: Any = None
atm_build_judge_prompt: Any = None


def load_atm_evaluator() -> None:
    """Load the pinned, MIT-licensed ATM-Bench evaluator subset."""

    global ATM_JUDGE_PROMPT, atm_number_core, atm_list_core, atm_build_judge_prompt
    vendor = REPO_ROOT / "eval/third_party/atm_bench"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from memqa.utils.evaluator.config import LLM_JUDGE_PROMPT
        from memqa.utils.evaluator.evaluate_qa import (
            _deterministic_accuracy_core,
            _list_jaccard_core,
            build_judge_prompt,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ATM-Bench scorer subset is missing or incomplete under "
            "eval/third_party/atm_bench."
        ) from exc
    ATM_JUDGE_PROMPT = LLM_JUDGE_PROMPT
    atm_number_core = _deterministic_accuracy_core
    atm_list_core = _list_jaccard_core
    atm_build_judge_prompt = build_judge_prompt


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memdaily_choice(prediction: str) -> str:
    """Extract one unambiguous A--D choice, ignoring answer-template wording."""

    text = str(prediction).strip()
    matches = {
        match.upper()
        for match in re.findall(r"(?i)(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", text)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return ""


def memdaily_official_prediction(prediction: str) -> str:
    """Match official MemDaily ``remove_space_and_ent`` normalization exactly."""

    return str(prediction).replace(" ", "").replace("\n", "")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def judge_once(client: JudgeClient, dataset: str, question: str, gold: str, prediction: str) -> dict[str, Any]:
    if dataset == "atm_bench_text_sgm":
        prompt = atm_build_judge_prompt(question, gold, prediction)
        data = client.chat_json([{"role": "user", "content": prompt}], max_tokens=600)
        return {
            "accuracy": 1.0 if parse_bool(data.get("accuracy")) else 0.0,
            "explanation": str(data.get("explanation", ""))[:1000],
            "raw_judge": data,
        }
    raise ValueError(f"judge not defined for dataset: {dataset}")


def judge_repeated(
    client: JudgeClient,
    dataset: str,
    question: str,
    gold: str,
    prediction: str,
    repeats: int,
    max_attempts: int,
    retry_sleep: float,
    retry_backoff: float,
) -> dict[str, Any]:
    attempts = []
    errors = []
    sleep_seconds = max(0.0, retry_sleep)
    for attempt_number in range(1, max_attempts + 1):
        try:
            attempts.append(judge_once(client, dataset, question, gold, prediction))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {"attempt": attempt_number, "type": type(exc).__name__, "message": str(exc)[:500]}
            )
            if len(attempts) < repeats and attempt_number < max_attempts and sleep_seconds > 0:
                time.sleep(min(sleep_seconds, 30.0))
                sleep_seconds *= retry_backoff
        if len(attempts) >= repeats:
            break
    if len(attempts) != repeats:
        raise RuntimeError(f"judge repeats incomplete: got {len(attempts)}/{repeats}; errors={errors}")
    accuracy_values = [float(item["accuracy"]) for item in attempts]
    result = {
        "judge_source": "openai_compatible_official_rubric_median",
        "judge_model": client.model,
        "judge_temperature": client.temperature,
        "judge_repeats": repeats,
        "judge_mode": "strict_only",
        "official_rubric": "ATM-Bench",
        "accuracy": float(statistics.median(accuracy_values)),
        "pass": sum(value >= 0.5 for value in accuracy_values) > repeats / 2,
        "attempt_accuracy": accuracy_values,
        "raw_judge_attempts": attempts,
    }
    if errors:
        result["judge_retry_errors"] = errors
    return result


def deterministic_score(instance: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    dataset = instance["dataset"]
    prediction = str(raw.get("raw_output", ""))
    gold = str(instance["answer"])
    if dataset == "atm_bench_text_sgm":
        qtype = instance["metadata"]["question_type"]
        if qtype == "number":
            correct, normalized = atm_number_core(
                gold, prediction, question=instance["metadata"]["original_question"]
            )
            return {"official_metric": "normalized_em", "official_score": float(correct), "normalized_prediction": normalized}
        if qtype == "list_recall":
            score, items = atm_list_core(gold, prediction)
            return {"official_metric": "jaccard", "official_score": float(score), "normalized_prediction": items}
        return {"official_metric": "llm_accuracy", "official_score": None}
    if dataset == "memdaily_official":
        official_prediction = memdaily_official_prediction(prediction)
        semantic_choice = memdaily_choice(prediction)
        return {
            "official_metric": "exact_choice_format_accuracy",
            "official_score": float(official_prediction == gold),
            "normalized_prediction": official_prediction,
            "format_compliant": official_prediction in {"A", "B", "C", "D"},
            "semantic_metric": "unique_choice_accuracy",
            "semantic_choice": semantic_choice,
            "semantic_score": float(semantic_choice == gold),
            "diagnostic_extracted_choice": semantic_choice,
        }
    raise ValueError(f"unsupported dataset: {dataset}")


def needs_judge(instance: dict[str, Any]) -> bool:
    return (
        instance["dataset"] == "atm_bench_text_sgm"
        and instance["metadata"]["question_type"] == "open_end"
    )


def score_row(
    instance: dict[str, Any],
    raw: dict[str, Any],
    client: JudgeClient | None,
    repeats: int,
    max_attempts: int,
    retry_sleep: float,
    retry_backoff: float,
) -> dict[str, Any]:
    deterministic = deterministic_score(instance, raw)
    judge = None
    if needs_judge(instance):
        if client is None or not client.available:
            raise RuntimeError("judge required but unavailable")
        judge = judge_repeated(
            client,
            instance["dataset"],
            instance["metadata"]["original_question"],
            str(instance["answer"]),
            str(raw.get("raw_output", "")),
            repeats,
            max_attempts,
            retry_sleep,
            retry_backoff,
        )
    if instance["dataset"] == "atm_bench_text_sgm" and needs_judge(instance):
        primary = judge["accuracy"] if judge else None
    elif instance["dataset"] == "memdaily_official":
        primary = deterministic["semantic_score"]
    else:
        primary = deterministic["official_score"]
    out = dict(raw)
    out["gold"] = {
        "answer": instance["answer"],
        "question_type": instance["metadata"]["question_type"],
        "original_question": instance["metadata"]["original_question"],
    }
    out["score"] = {
        **deterministic,
        "primary_score": primary,
        "official_judge": judge,
    }
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    values = []
    judge_failures = 0
    for row in rows:
        value = row["score"].get("primary_score")
        if value is None or not math.isfinite(float(value)):
            judge_failures += 1
            continue
        value = float(value)
        values.append(value)
        by_type[str(row["gold"]["question_type"])].append(value)
    return {
        "count": len(rows),
        "scored_count": len(values),
        "primary_score": sum(values) / len(values) if values else None,
        "by_question_type": {
            key: {"count": len(items), "primary_score": sum(items) / len(items)}
            for key, items in sorted(by_type.items())
        },
        "judge_failure_count": judge_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--judge-max-attempts", type=int, default=6)
    parser.add_argument("--judge-retry-sleep", type=float, default=0.0)
    parser.add_argument("--judge-retry-backoff", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--judge-base-url", default="https://api.openai.com")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only the leading N frozen instances; 0 requires the full dataset.",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.judge_repeats != 3:
        raise ValueError("frozen scorer requires concurrency >=1 and judge_repeats=3")
    if args.judge_max_attempts < args.judge_repeats:
        raise ValueError("judge_max_attempts must be >= repeats")
    if args.judge_retry_sleep < 0 or args.judge_retry_backoff < 1:
        raise ValueError("judge retry sleep must be >=0 and backoff must be >=1")

    load_dotenv(REPO_ROOT / ".env")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    instances = read_jsonl(args.instances)
    if args.limit:
        instances = instances[: args.limit]
    if any(row.get("dataset") == "atm_bench_text_sgm" for row in instances):
        load_atm_evaluator()
    raw_rows = read_jsonl(args.input)
    instance_by_id = {row["instance_id"]: row for row in instances}
    if len(raw_rows) != len(instances):
        raise ValueError(f"raw/instance row mismatch: {len(raw_rows)} != {len(instances)}")
    raw_ids = [row["instance_id"] for row in raw_rows]
    if raw_ids != [row["instance_id"] for row in instances]:
        raise ValueError("raw order/id coverage differs from frozen instances")

    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = {row["instance_id"]: row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("duplicate instance id in resumable scored output")
    pending = [(instance_by_id[row["instance_id"]], row) for row in raw_rows if row["instance_id"] not in existing]
    score_meta_path = args.output.with_suffix(".score_meta.json")
    previous_meta = (
        json.loads(score_meta_path.read_text(encoding="utf-8"))
        if score_meta_path.is_file()
        else {}
    )

    client = None
    api_status = None
    judge_required = any(needs_judge(instance) for instance in instances)
    resumed_judge_model = previous_meta.get("judge_model")
    if any(needs_judge(instance) for instance, _ in pending):
        client = JudgeClient(
            args.judge_base_url,
            os.environ.get(args.api_key_env),
            args.judge_model,
            args.timeout,
            0.0,
        )
        api_status = client.check()
        if not api_status.get("available"):
            raise RuntimeError(f"Judge endpoint/model check failed: {api_status}")
    elif judge_required:
        api_status = previous_meta.get("judge_api_status")
        if not (api_status or {}).get("available"):
            judged_rows = [
                row
                for row in existing_rows
                if (row.get("score") or {}).get("official_judge")
            ]
            if not judged_rows:
                raise RuntimeError(
                    "resumed ATM output has no pending judge rows and no prior judge provenance"
                )
            models = {
                row["score"]["official_judge"].get("judge_model")
                for row in judged_rows
                if row["score"]["official_judge"].get("judge_model")
            }
            if len(models) != 1:
                raise RuntimeError(
                    f"resumed ATM output has ambiguous judge models: {sorted(models)}"
                )
            resumed_judge_model = models.pop()
            api_status = {
                "available": True,
                "source": "completed_judge_rows",
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    started = time.time()

    def persist(row: dict[str, Any]) -> None:
        with write_lock:
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            existing[row["instance_id"]] = row

    if args.concurrency == 1:
        for instance, raw in pending:
            persist(
                score_row(
                    instance,
                    raw,
                    client,
                    args.judge_repeats,
                    args.judge_max_attempts,
                    args.judge_retry_sleep,
                    args.judge_retry_backoff,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    score_row,
                    instance,
                    raw,
                    client,
                    args.judge_repeats,
                    args.judge_max_attempts,
                    args.judge_retry_sleep,
                    args.judge_retry_backoff,
                ): raw["instance_id"]
                for instance, raw in pending
            }
            completed = 0
            for future in as_completed(futures):
                persist(future.result())
                completed += 1
                if completed % 50 == 0 or completed == len(futures):
                    print(json.dumps({"event": "score_progress", "completed": completed, "pending": len(futures)}), flush=True)

    ordered = [existing[row["instance_id"]] for row in raw_rows]
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instances": str(args.instances),
        "instances_sha256": sha256_file(args.instances),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "row_count": len(ordered),
        "first_instance_id": ordered[0]["instance_id"],
        "last_instance_id": ordered[-1]["instance_id"],
        "strict_only": True,
        "judge_repeats": args.judge_repeats,
        "judge_concurrency": args.concurrency,
        "judge_max_attempts": args.judge_max_attempts,
        "judge_retry_sleep": args.judge_retry_sleep,
        "judge_retry_backoff": args.judge_retry_backoff,
        "scorer_commit": os.environ.get("MEMQA_OOD_SCORER_COMMIT"),
        "scorer_code_sha256": sha256_file(Path(__file__)),
        "judge_temperature": 0.0,
        "judge_required": judge_required,
        "judge_model": (
            args.judge_model
            if client
            else resumed_judge_model
            if judge_required
            else None
        ),
        "judge_api_status": (
            api_status
            if judge_required
            else {"required": False, "available": None}
        ),
        "judge_failure_count": summary["judge_failure_count"],
        "official_code": {
            "atm_revision": "d463445614ad78a48736b98ab901795f7ecaf3da",
            "atm_prompt_sha256": hashlib.sha256(ATM_JUDGE_PROMPT.encode()).hexdigest(),
            "atm_vendor_manifest_sha256": sha256_file(ATM_VENDOR_MANIFEST),
        },
        "judge_substitution": (
            "official rubric evaluated by the configured OpenAI-compatible judge"
            if judge_required
            else None
        ),
        "elapsed_sec": round(time.time() - started, 3),
        "summary": summary,
    }
    if summary["judge_failure_count"]:
        raise RuntimeError(f"nonzero judge failure count: {summary['judge_failure_count']}")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
