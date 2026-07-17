#!/usr/bin/env python3
"""Score MemQA raw outputs with normalized F1/EM and optional LLM judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval.common.judge import DEFAULT_BASE_URL, DEFAULT_MODEL, JudgeClient

DEFAULT_JUDGE_MODEL = DEFAULT_MODEL
NORMAL_JUDGE_MODE = "normal"
STRICT_JUDGE_MODE = "strict"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(text: Any) -> str:
    s = str(text).lower().replace(",", "")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: Any, gold: Any) -> bool:
    return set(normalize_answer(prediction).split()) == set(normalize_answer(gold).split())


def f1_score(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_prompt(mode: str) -> tuple[str, str]:
    if mode == STRICT_JUDGE_MODE:
        system = (
            "You are a conservative but fair evaluator for a memory question-answering benchmark. "
            "Your job is to avoid overly generous partial credit while still accepting truly "
            "equivalent answers, aliases, abbreviations, and harmless formatting differences. "
            "Return JSON only."
        )
        instruction = (
            "Grade model_output against gold_answer for the question. Return JSON with keys: "
            "score (0 to 1), pass (boolean), matched_points (array of strings), "
            "missed_points (array of strings), and rationale (short string). Use this strict "
            "rubric: give 1.0 only when the answer contains the correct core entity/value/date/"
            "relationship asked for, allowing aliases and semantically equivalent wording. Give "
            "0.5 to 0.75 only when the output includes the correct core answer but has minor "
            "extra wording, minor imprecision, or one secondary omission. Give 0 for a different "
            "person, organization, place, number, date, title, relation, or answer choice; for "
            "a broad category when the gold answer is a specific entity; for answers that merely "
            "share common words with gold; for plausible guesses unsupported by the exact answer; "
            "or when the model says the answer is unknown/unavailable while gold is answerable. "
            "Do not reward explanation quality if the final answer is wrong. If the question asks "
            "for a country/state/type and the model gives exactly that correct country/state/type, "
            "it is correct even if it could be guessed from world knowledge."
        )
        return system, instruction
    system = (
        "You are a strict but fair evaluator for a memory question-answering benchmark. "
        "Accept semantically equivalent short answers. Do not require exact wording. "
        "Penalize hallucinated entities, wrong dates, wrong relationships, and answers "
        "that contradict the gold answer. Return JSON only."
    )
    instruction = (
        "Grade model_output against gold_answer for the question. Return JSON with keys: "
        "score (0 to 1), pass (boolean), matched_points (array of strings), "
        "missed_points (array of strings), and rationale (short string). Use partial "
        "credit only when the answer is partly correct. If the model says the answer "
        "is unknown or unavailable when the gold answer is answerable, score 0."
    )
    return system, instruction


def judge_once(client: JudgeClient, payload: dict[str, Any], mode: str = NORMAL_JUDGE_MODE) -> dict[str, Any]:
    system, instruction = judge_prompt(mode)
    user = {"instruction": instruction, **payload}
    data = client.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        max_tokens=384,
    )
    score = max(0.0, min(1.0, float(data.get("score", 1.0 if data.get("pass") else 0.0))))
    return {
        "score": score,
        "pass": bool(data.get("pass", score >= 0.75)),
        "judge_mode": mode,
        "matched_points": data.get("matched_points", []),
        "missed_points": data.get("missed_points", []),
        "rationale": str(data.get("rationale", ""))[:1000],
        "raw_judge": data,
    }


def judge_median(
    client: JudgeClient,
    payload: dict[str, Any],
    repeats: int,
    mode: str = NORMAL_JUDGE_MODE,
    max_attempts: int | None = None,
    retry_sleep: float = 0.0,
    retry_backoff: float = 1.0,
) -> dict[str, Any]:
    attempts = []
    errors = []
    max_total = max(repeats, max_attempts or repeats * 2)
    sleep_seconds = max(0.0, retry_sleep)
    backoff = max(1.0, retry_backoff)
    tries = 0
    while len(attempts) < repeats and tries < max_total:
        tries += 1
        try:
            attempts.append(judge_once(client, payload, mode))
        except Exception as exc:  # noqa: BLE001
            errors.append({"attempt": tries, "error_type": type(exc).__name__, "message": str(exc)[:500]})
            if len(attempts) < repeats and tries < max_total and sleep_seconds > 0:
                time.sleep(min(sleep_seconds, 30.0))
                sleep_seconds *= backoff
    if not attempts:
        return {
            "judge_source": "api_error",
            "judge_mode": mode,
            "score": None,
            "pass": False,
            "judge_attempts_requested": repeats,
            "judge_attempts_total": tries,
            "judge_max_attempts": max_total,
            "judge_errors": errors,
        }
    scores = [float(item["score"]) for item in attempts]
    median = float(statistics.median(scores))
    selected = min(attempts, key=lambda item: abs(float(item["score"]) - median))
    out = {
        "judge_source": "api_median",
        "judge_mode": mode,
        "judge_model": client.model,
        "judge_temperature": client.temperature,
        "judge_repeats": len(attempts),
        "judge_attempts_requested": repeats,
        "judge_attempts_total": tries,
        "judge_max_attempts": max_total,
        "score": median,
        "pass": sum(1 for item in attempts if item.get("pass")) > len(attempts) / 2,
        "attempt_scores": scores,
        "score_min": min(scores),
        "score_max": max(scores),
        "score_range": max(scores) - min(scores),
        "matched_points": selected.get("matched_points", []),
        "missed_points": selected.get("missed_points", []),
        "rationale": selected.get("rationale", ""),
        "raw_judge_attempts": attempts,
    }
    if errors:
        out["judge_errors"] = errors
    return out


def has_judge_failures(meta: dict[str, Any]) -> bool:
    if not meta.get("judge_api_status", {}).get("available"):
        return True
    failed_sources = {"api_error", "unavailable", "missing"}
    if not meta.get("strict_only") and any(meta.get("judge_sources", {}).get(source, 0) for source in failed_sources):
        return True
    if meta.get("strict_judge") and any(meta.get("strict_judge_sources", {}).get(source, 0) for source in failed_sources):
        return True
    return False


def score_one_record(
    raw: dict[str, Any],
    instance: dict[str, Any],
    client: JudgeClient,
    api_status: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = raw.get("raw_output", "")
    gold = instance.get("answer", "")
    deterministic = {
        "normalized_f1": round(f1_score(output, gold), 6),
        "exact_match": exact_match(output, gold),
    }
    payload = {
        "question": instance.get("question"),
        "gold_answer": gold,
        "model_output": output,
        "raw_category": instance.get("metadata", {}).get("raw_category"),
        "is_adversarial": instance.get("metadata", {}).get("is_adversarial"),
        "baseline": raw.get("baseline"),
    }
    if args.strict_only:
        judged = {
            "judge_source": "skipped_strict_only",
            "judge_mode": NORMAL_JUDGE_MODE,
            "score": None,
            "pass": False,
            "rationale": "Normal judge skipped because --strict-only was requested.",
        }
        strict_judged = (
            judge_median(
                client,
                payload,
                args.judge_repeats,
                STRICT_JUDGE_MODE,
                max_attempts=args.judge_max_attempts or None,
                retry_sleep=args.judge_retry_sleep,
                retry_backoff=args.judge_retry_backoff,
            )
            if client.available
            else {
                "judge_source": "unavailable",
                "judge_mode": STRICT_JUDGE_MODE,
                "score": None,
                "pass": False,
                "rationale": "Judge API unavailable; deterministic metrics only.",
            }
        )
    elif client.available:
        judged = judge_median(
            client,
            payload,
            args.judge_repeats,
            max_attempts=args.judge_max_attempts or None,
            retry_sleep=args.judge_retry_sleep,
            retry_backoff=args.judge_retry_backoff,
        )
        strict_judged = (
            judge_median(
                client,
                payload,
                args.judge_repeats,
                STRICT_JUDGE_MODE,
                max_attempts=args.judge_max_attempts or None,
                retry_sleep=args.judge_retry_sleep,
                retry_backoff=args.judge_retry_backoff,
            )
            if args.strict_judge
            else None
        )
    else:
        judged = {
            "judge_source": "unavailable",
            "judge_mode": NORMAL_JUDGE_MODE,
            "score": None,
            "pass": False,
            "rationale": "Judge API unavailable; deterministic metrics only.",
        }
        strict_judged = (
            {
                "judge_source": "unavailable",
                "judge_mode": STRICT_JUDGE_MODE,
                "score": None,
                "pass": False,
                "rationale": "Judge API unavailable; deterministic metrics only.",
            }
            if args.strict_judge
            else None
        )
    row = dict(raw)
    row["gold"] = {
        "answer": gold,
        "evidence": instance.get("evidence", []),
        "raw_category": instance.get("metadata", {}).get("raw_category"),
    }
    row["score"] = {
        **deterministic,
        "llm_judge_score": judged.get("score"),
        "llm_judge_pass": judged.get("pass"),
        "judge": judged,
    }
    if strict_judged is not None:
        row["score"].update(
            {
                "llm_judge_strict_score": strict_judged.get("score"),
                "llm_judge_strict_pass": strict_judged.get("pass"),
                "strict_judge": strict_judged,
            }
        )
    row["judge_api_status"] = api_status
    return row


def log_progress(completed: int, total: int, started: float, args: argparse.Namespace) -> None:
    if not args.progress_every:
        return
    if completed != total and completed % args.progress_every != 0:
        return
    elapsed = max(time.time() - started, 1e-6)
    print(
        json.dumps(
            {
                "event": "score_progress",
                "completed": completed,
                "total": total,
                "elapsed_sec": round(elapsed, 3),
                "records_per_min": round(completed / elapsed * 60, 3),
                "concurrency": args.concurrency,
                "judge_repeats": args.judge_repeats,
                "strict_judge": args.strict_judge,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


def score_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instances = {item["instance_id"]: item for item in read_jsonl(args.instances)}
    raw_records = read_jsonl(args.input)
    client = JudgeClient(args.judge_base_url, os.environ.get(args.api_key_env), args.judge_model, args.timeout, args.judge_temperature)
    api_status = client.check()
    scored: list[dict[str, Any] | None] = [None] * len(raw_records)
    judge_sources: Counter[str] = Counter()
    strict_judge_sources: Counter[str] = Counter()
    started = time.time()

    if args.concurrency <= 1 or len(raw_records) <= 1:
        for index, raw in enumerate(raw_records):
            row = score_one_record(raw, instances[raw["instance_id"]], client, api_status, args)
            scored[index] = row
            judge_sources[row["score"]["judge"].get("judge_source", "missing")] += 1
            if "strict_judge" in row["score"]:
                strict_judge_sources[row["score"]["strict_judge"].get("judge_source", "missing")] += 1
            log_progress(index + 1, len(raw_records), started, args)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(score_one_record, raw, instances[raw["instance_id"]], client, api_status, args): index
                for index, raw in enumerate(raw_records)
            }
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                row = future.result()
                scored[index] = row
                judge_sources[row["score"]["judge"].get("judge_source", "missing")] += 1
                if "strict_judge" in row["score"]:
                    strict_judge_sources[row["score"]["strict_judge"].get("judge_source", "missing")] += 1
                completed += 1
                log_progress(completed, len(raw_records), started, args)

    final_scored = [row for row in scored if row is not None]
    meta = {
        "input": str(args.input),
        "instances": str(args.instances),
        "output": str(args.output),
        "records": len(final_scored),
        "judge_api_status": api_status,
        "judge_sources": dict(judge_sources),
        "strict_judge": args.strict_judge,
        "strict_only": args.strict_only,
        "strict_judge_sources": dict(strict_judge_sources),
        "judge_repeats": args.judge_repeats,
        "judge_max_attempts": args.judge_max_attempts or args.judge_repeats * 2,
        "judge_retry_sleep": args.judge_retry_sleep,
        "judge_retry_backoff": args.judge_retry_backoff,
        "fail_on_judge_error": args.fail_on_judge_error,
        "concurrency": args.concurrency,
        "elapsed_sec": round(time.time() - started, 3),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return final_scored, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument("--strict-judge", action="store_true", help="Also run a stricter LLM judge rubric and write llm_judge_strict_* fields.")
    parser.add_argument("--strict-only", action="store_true", help="Run only the strict LLM judge; skip normal judge API calls.")
    parser.add_argument("--judge-max-attempts", type=int, default=0, help="Maximum judge attempts per record/mode; 0 uses repeats*2.")
    parser.add_argument("--judge-retry-sleep", type=float, default=0.0, help="Initial sleep in seconds after a failed judge attempt.")
    parser.add_argument("--judge-retry-backoff", type=float, default=1.0, help="Multiplier for judge retry sleeps; values below 1 are rejected.")
    parser.add_argument("--fail-on-judge-error", action="store_true", help="Do not write scored output if any judge result is unavailable or api_error.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of records to score concurrently. Default preserves serial scoring.")
    parser.add_argument("--progress-every", type=int, default=50, help="Emit stderr progress every N completed records; 0 disables progress logs.")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.judge_repeats < 1:
        raise ValueError("--judge-repeats must be >= 1")
    if args.strict_only:
        args.strict_judge = True
    if args.judge_max_attempts < 0:
        raise ValueError("--judge-max-attempts must be >= 0")
    if args.judge_max_attempts and args.judge_max_attempts < args.judge_repeats:
        raise ValueError("--judge-max-attempts must be >= --judge-repeats")
    if args.judge_retry_sleep < 0:
        raise ValueError("--judge-retry-sleep must be >= 0")
    if args.judge_retry_backoff < 1:
        raise ValueError("--judge-retry-backoff must be >= 1")
    scored, meta = score_records(args)
    if args.fail_on_judge_error and has_judge_failures(meta):
        failure_path = args.output.with_suffix(".failed_score_meta.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            "judge_failed; "
            f"wrote failure meta to {failure_path}; "
            f"judge_sources={meta.get('judge_sources')}; "
            f"strict_judge_sources={meta.get('strict_judge_sources')}"
        )
    write_jsonl(args.output, scored)
    args.output.with_suffix(".score_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
