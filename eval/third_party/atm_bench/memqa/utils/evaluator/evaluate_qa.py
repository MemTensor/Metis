#!/usr/bin/env python3
"""Evaluate QA predictions with deterministic, LLM-judge, and ATM metrics."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Optional for the vendored deterministic metric subset.
    import requests
except ImportError:  # pragma: no cover - full upstream CLI only
    requests = None

try:  # Optional for the vendored deterministic metric subset.
    from tqdm import tqdm
except ImportError:  # pragma: no cover - full upstream CLI only
    def tqdm(iterable, *args, **kwargs):
        return iterable

from memqa.utils.evaluator.config import EVALUATOR_CONFIG, LLM_JUDGE_PROMPT
from memqa.utils.evaluator.normalizer import (
    extract_codes,
    extract_dates,
    extract_currency_amounts,
    extract_numbers,
    extract_reference_date,
    extract_times,
    is_abstention,
    location_token_match,
    normalize_currency_codes,
    normalize_text,
    remove_date_time_text,
    resolve_relative_dates,
    strip_context_phrases,
    split_list_items,
    token_subset_match,
    strip_parenthetical_details,
    strip_currency_breakdowns,
    normalize_between_to_range,
    strip_leading_articles,
    aggressive_preprocess,
    semantic_units_match,
)
from memqa.utils.evaluator.qtype_utils import (
    QTYPE_LIST,
    QTYPE_NUMBER,
    QTYPE_OPEN,
    detect_qtype,
)

# Memory item ID patterns: YYYYMMDD_HHMMSS (image/video) and emailXXXX (email)
_RE_MEDIA_ID = re.compile(r"\b(\d{8}_\d{6})\b")
_RE_EMAIL_ID = re.compile(r"\b(email\d+)\b", re.IGNORECASE)


def _extract_list_ids_from_text(text: str) -> list[str]:
    """Extract memory item IDs from free-text prose.

    Applied uniformly to *all* prediction sources during list_jaccard scoring
    so that agent systems, baselines, and any other pipeline are treated
    identically regardless of whether their answers are structured or prose.
    """
    if not text:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for m in _RE_MEDIA_ID.finditer(text):
        v = m.group(1)
        if v not in seen:
            seen.add(v)
            ids.append(v)
    for m in _RE_EMAIL_ID.finditer(text):
        v = m.group(1).lower()
        if v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_qa_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "qas" in data:
        qas = data["qas"]
        if isinstance(qas, list):
            return qas
    raise ValueError("Unsupported QA schema. Expected list or dict with 'qas'.")


def write_json_list(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        if isinstance(data, list):
            return {str(item.get("id")): item for item in data if "id" in item}
        return {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def write_incremental_result(
    path: Path, new_result: Dict[str, Any], existing_results: Dict[str, Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qa_id = str(new_result.get("id"))
    existing_results[qa_id] = new_result
    all_results = list(existing_results.values())
    with path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def date_component_match(gt: str, pred: str) -> bool:
    if len(gt) == 4 or len(pred) == 4:
        return gt[-4:] == pred[-4:]
    return gt == pred


def date_token_match(gt_token: str, pred_token: str) -> bool:
    if "-" in gt_token:
        if "-" not in pred_token:
            return False
        gt_start, gt_end = gt_token.split("-", 1)
        pred_start, pred_end = pred_token.split("-", 1)
        return date_component_match(gt_start, pred_start) and date_component_match(
            gt_end, pred_end
        )
    return date_component_match(gt_token, pred_token)


def dates_match(gt_tokens: List[str], pred_tokens: List[str]) -> bool:
    remaining = list(pred_tokens)
    for gt_token in gt_tokens:
        matched = False
        for idx, pred_token in enumerate(remaining):
            if date_token_match(gt_token, pred_token):
                matched = True
                remaining.pop(idx)
                break
        if not matched:
            return False
    return True


def tokens_match(expected: List, actual: List) -> bool:
    return Counter(expected) == Counter(actual)


def _deterministic_accuracy_core(
    ground_truth: str, prediction: str, question: Optional[str] = None
) -> tuple[bool, str]:
    is_abst_gt = is_abstention(ground_truth)
    is_abst_pred = is_abstention(prediction)

    reference_date = extract_reference_date(question or "")
    ground_truth_resolved = resolve_relative_dates(ground_truth, reference_date)
    prediction_resolved = resolve_relative_dates(prediction, reference_date)

    ground_truth_prep = aggressive_preprocess(ground_truth_resolved)
    prediction_prep = aggressive_preprocess(prediction_resolved)

    ground_truth_clean = strip_parenthetical_details(ground_truth_prep)
    prediction_clean = strip_parenthetical_details(prediction_prep)

    ground_truth_clean = strip_currency_breakdowns(ground_truth_clean)
    prediction_clean = strip_currency_breakdowns(prediction_clean)

    ground_truth_clean = normalize_between_to_range(ground_truth_clean)
    prediction_clean = normalize_between_to_range(prediction_clean)

    ground_truth_clean = strip_leading_articles(ground_truth_clean)
    prediction_clean = strip_leading_articles(prediction_clean)

    gt_normalized = normalize_text(ground_truth_clean)
    pred_normalized = normalize_text(prediction_clean)

    if is_abst_gt or is_abst_pred:
        return (is_abst_gt and is_abst_pred, pred_normalized)

    gt_codes = extract_codes(ground_truth_clean)
    if gt_codes:
        pred_upper = prediction_clean.upper()
        pred_stripped = re.sub(
            r"\b(?:CODE|ID|REFERENCE|REF|NUMBER|NUM|LABEL):\s*", "", pred_upper
        )
        if not all(code in pred_stripped for code in gt_codes):
            return False, pred_normalized

    gt_dates = [token.value for token in extract_dates(ground_truth_clean)]
    pred_dates = [token.value for token in extract_dates(prediction_clean)]

    if gt_dates:
        if not pred_dates:
            return False, pred_normalized
        if not dates_match(gt_dates, pred_dates):
            return False, pred_normalized

    gt_times = [token.value for token in extract_times(ground_truth_clean)]
    pred_times = [token.value for token in extract_times(prediction_clean)]
    if gt_times and pred_times:
        if not tokens_match(gt_times, pred_times):
            return False, pred_normalized

    if gt_dates or gt_times:
        gt_remainder = normalize_text(remove_date_time_text(ground_truth_clean))
        pred_remainder = normalize_text(remove_date_time_text(prediction_clean))

        gt_remainder_stripped = strip_context_phrases(gt_remainder)
        pred_remainder_stripped = strip_context_phrases(pred_remainder)

        if not gt_remainder_stripped:
            return True, pred_normalized
        if token_subset_match(gt_remainder_stripped, pred_remainder_stripped):
            return True, pred_normalized

    gt_clean = remove_date_time_text(ground_truth_clean)
    pred_clean = remove_date_time_text(prediction_clean)
    gt_numbers, gt_currencies = extract_numbers(gt_clean)
    pred_numbers, pred_currencies = extract_numbers(pred_clean)
    gt_currency_amounts = extract_currency_amounts(gt_clean)
    pred_currency_amounts = extract_currency_amounts(pred_clean)

    if gt_currencies:
        gt_norm = normalize_currency_codes(gt_currencies)
        pred_norm = normalize_currency_codes(pred_currencies)
        if pred_norm and not tokens_match(gt_norm, pred_norm):
            return False, pred_normalized

    if gt_currency_amounts:
        if not pred_currency_amounts:
            return False, pred_normalized
        if not tokens_match(gt_currency_amounts, pred_currency_amounts):
            return False, pred_normalized
        gt_numbers = [num for num in gt_numbers if num not in gt_currency_amounts]
        pred_numbers = [num for num in pred_numbers if num not in pred_currency_amounts]

    if gt_numbers:
        if not pred_numbers:
            return False, pred_normalized
        if not tokens_match(gt_numbers, pred_numbers):
            return False, pred_normalized

    gt_items = split_list_items(ground_truth_clean)
    if len(gt_items) >= 2:
        if not all(item in pred_normalized for item in gt_items):
            return False, pred_normalized
        return True, pred_normalized

    if gt_normalized.startswith("yes") or gt_normalized.startswith("no"):
        if pred_normalized in {"yes", "no"}:
            return gt_normalized.startswith(pred_normalized), pred_normalized

    if location_token_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    if len(gt_normalized) >= 3 and gt_normalized in pred_normalized:
        return True, pred_normalized

    if len(pred_normalized) >= 3 and pred_normalized in gt_normalized:
        return True, pred_normalized

    if token_subset_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    # Final fallback: semantic units match
    if semantic_units_match(ground_truth_clean, prediction_clean):
        return True, pred_normalized

    return gt_normalized == pred_normalized, pred_normalized


def deterministic_accuracy(
    ground_truth: str, prediction: str, question: Optional[str] = None
) -> bool:
    return _deterministic_accuracy_core(ground_truth, prediction, question)[0]


def _prediction_list_items(prediction: str) -> list[str]:
    pred_ids = _extract_list_ids_from_text(prediction)
    if pred_ids:
        return pred_ids
    return [item for item in split_list_items(prediction) if item]


def _list_jaccard_core(ground_truth: str, prediction: str) -> tuple[float, list[str]]:
    gt_items = {item for item in split_list_items(ground_truth) if item}
    pred_items_list = _prediction_list_items(prediction)
    pred_items = set(pred_items_list)
    if not gt_items and not pred_items:
        return 1.0, pred_items_list
    if not gt_items:
        return 0.0, pred_items_list
    union = gt_items | pred_items
    if not union:
        return 0.0, pred_items_list
    return len(gt_items & pred_items) / len(union), pred_items_list


def list_jaccard_score(ground_truth: str, prediction: str) -> float:
    return _list_jaccard_core(ground_truth, prediction)[0]


class JudgeResponseError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class OpenAIEmptyResponseError(JudgeResponseError):
    def __init__(self, message: str = "Empty OpenAI response output_text"):
        super().__init__(message, retryable=True)


class OpenAIRefusalError(JudgeResponseError):
    def __init__(self, refusal_text: str):
        message = refusal_text.strip() or "OpenAI judge refused to answer"
        super().__init__(message, retryable=False)
        self.refusal_text = message


def _coerce_openai_obj(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        dumped = {
            key: val for key, val in vars(value).items() if not str(key).startswith("_")
        }
        if isinstance(dumped, dict):
            return dumped
    return {}


def _extract_openai_response_text_and_refusal(response: Any) -> Tuple[str, str]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        text = str(output_text).strip()
        if text:
            return text, ""

    texts: List[str] = []
    refusals: List[str] = []

    def collect_refusal(value: Any) -> None:
        if isinstance(value, str):
            refusal_text = value.strip()
            if refusal_text:
                refusals.append(refusal_text)
        elif isinstance(value, list):
            for item in value:
                collect_refusal(item)
        elif value is not None:
            refusal_text = str(value).strip()
            if refusal_text:
                refusals.append(refusal_text)

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        output = [output] if output is not None else []

    for item in output:
        item_dict = _coerce_openai_obj(item)
        if not item_dict:
            continue
        if item_dict.get("type") == "refusal":
            collect_refusal(
                item_dict.get("refusal")
                or item_dict.get("text")
                or item_dict.get("content")
            )

        content = item_dict.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
        elif isinstance(content, list):
            for part in content:
                part_dict = _coerce_openai_obj(part)
                if not part_dict:
                    continue
                part_type = str(part_dict.get("type", ""))
                part_text = part_dict.get("text")
                if part_type in {"output_text", "text"} and part_text:
                    texts.append(str(part_text).strip())
                if part_type == "refusal":
                    collect_refusal(
                        part_dict.get("refusal")
                        or part_dict.get("text")
                        or part_dict.get("content")
                    )
                elif part_dict.get("refusal") is not None:
                    collect_refusal(part_dict.get("refusal"))

        if item_dict.get("refusal") is not None:
            collect_refusal(item_dict.get("refusal"))

    if texts:
        return "\n".join(text for text in texts if text).strip(), ""

    refusal_text = "\n".join(text for text in refusals if text).strip()
    return "", refusal_text


class EvaluatorLLM:
    def __init__(self, provider: str, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self.openai_client: Optional[Any] = None
        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required for --judge-provider openai."
                ) from exc
            self.openai_client = OpenAI(api_key=self.config.get("api_key"))

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        if self.provider == "openai":
            return self._chat_openai(messages)
        if self.provider == "vllm":
            return self._chat_vllm_http(messages)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(self, messages: List[Dict[str, Any]]) -> str:
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        model = self.config.get("model")
        if not model:
            raise ValueError("Model is required for OpenAI provider")
        max_tokens_value = self.config.get("max_tokens")
        is_newer_model = any(x in model.lower() for x in ["gpt-5", "o1", "o3", "o4"])

        input_text = ""
        if len(messages) == 1 and messages[0].get("role") == "user":
            content = messages[0].get("content", "")
            input_text = content if isinstance(content, str) else ""
        if not input_text:
            lines = []
            for message in messages:
                role = message.get("role", "user").upper()
                content = message.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                lines.append(f"{role}: {content}")
            lines.append("ASSISTANT:")
            input_text = "\n".join(lines)

        kwargs: Dict[str, Any] = {
            "model": model,
            "input": input_text,
        }
        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if max_tokens_value:
            kwargs["max_output_tokens"] = max_tokens_value
        if self.config.get("temperature") is not None and not is_newer_model:
            kwargs["temperature"] = self.config.get("temperature")

        response = self.openai_client.responses.create(**kwargs)
        output_text, refusal_text = _extract_openai_response_text_and_refusal(response)
        if output_text:
            return output_text
        if refusal_text:
            print(
                f"Warning: OpenAI judge refusal detected: {refusal_text}",
                file=sys.stderr,
            )
            raise OpenAIRefusalError(refusal_text)
        print("Warning: empty OpenAI response output_text", file=sys.stderr)
        raise OpenAIEmptyResponseError()

    def _chat_vllm_http(self, messages: List[Dict[str, Any]]) -> str:
        endpoint = self.config.get("endpoint")
        if not endpoint:
            raise ValueError("VLLM endpoint is required for vllm provider")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get('api_key')}",
        }
        data = {
            "model": self.config.get("model"),
            "messages": messages,
            "max_tokens": self.config.get("max_tokens"),
            "temperature": self.config.get("temperature", 0.0),
        }

        # Add thinking mode control if specified (for GLM-4.7 and similar models)
        thinking_mode = self.config.get("thinking_mode")
        if thinking_mode:
            data["thinking"] = {"type": thinking_mode}

        response = requests.post(
            endpoint,
            headers=headers,
            json=data,
            timeout=self.config.get("timeout"),
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


def build_judge_prompt(question: str, answer: str, prediction: str) -> str:
    prompt = LLM_JUDGE_PROMPT.replace("{{question}}", question)
    prompt = prompt.replace("{{answer}}", answer)
    prompt = prompt.replace("{{prediction}}", prediction)
    return prompt


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(raw_text[start : end + 1])
            if "accuracy" in payload:
                return payload
        except json.JSONDecodeError:
            pass

    raw_lower = raw_text.lower()

    if '"accuracy": true' in raw_lower or '"accuracy":true' in raw_lower:
        accuracy_value = True
    elif '"accuracy": false' in raw_lower or '"accuracy":false' in raw_lower:
        accuracy_value = False
    elif any(
        phrase in raw_lower
        for phrase in ["accurate", "correct", "matches", "true", "yes"]
    ):
        accuracy_value = True
    else:
        accuracy_value = False

    explanation = raw_text if len(raw_text) < 500 else raw_text[:500] + "..."

    return {
        "accuracy": accuracy_value,
        "explanation": explanation,
        "parse_fallback": True,
    }


def run_deterministic(
    qas: List[Dict[str, Any]], predictions: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    correct = 0
    for qa in tqdm(qas, desc="Deterministic eval", unit="qa"):
        qa_id = str(qa.get("id"))
        prediction = predictions.get(qa_id, "")
        ground_truth = str(qa.get("answer", ""))
        qtype = normalize_qtype_value(qa.get("qtype"), ground_truth)
        normalized_prediction = None
        if qtype == QTYPE_NUMBER:
            is_correct, normalized_prediction = _deterministic_accuracy_core(
                ground_truth,
                prediction,
                question=str(qa.get("question", "")),
            )
        else:
            is_correct = deterministic_accuracy(
                ground_truth, prediction, question=str(qa.get("question", ""))
            )
            if qtype == QTYPE_LIST:
                normalized_prediction = ", ".join(_prediction_list_items(prediction))
        row = {
            "id": qa_id,
            "question": qa.get("question"),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "qtype": qtype,
            "accuracy": is_correct,
        }
        if normalized_prediction is not None:
            row["normalized_prediction"] = normalized_prediction
        rows.append(row)
        correct += int(is_correct)

    summary = {
        "count": len(qas),
        "correct": correct,
        "accuracy": correct / len(qas) if qas else 0.0,
        "by_qtype": summarize_atm(rows)["by_qtype"],
    }
    return rows, summary


def run_llm_judge(
    qas: List[Dict[str, Any]],
    predictions: Dict[str, str],
    provider: str,
    config: Dict[str, Any],
    max_workers: int,
    request_delay: float = 0.0,
    output_path: Optional[Path] = None,
    max_retries: int = 3,
    seed_results: Optional[Dict[str, Dict[str, Any]]] = None,
    finalize_output: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    thread_local = threading.local()
    write_lock = threading.Lock()
    qtype_by_id = {
        str(qa.get("id")): normalize_qtype_value(qa.get("qtype"), str(qa.get("answer", "")))
        for qa in qas
    }

    existing_results = {}
    failed_results: Dict[str, Dict[str, Any]] = {}
    if output_path and output_path.exists():
        existing_results = load_existing_results(output_path)
        print(
            f"Loaded {len(existing_results)} existing results, resuming from checkpoint..."
        )
    if seed_results:
        for qa_id, result in seed_results.items():
            if qa_id not in existing_results:
                existing_results[qa_id] = result
    if existing_results:
        for qa_id, result in list(existing_results.items()):
            if result.get("failed"):
                failed_results[qa_id] = result
                existing_results.pop(qa_id, None)

    cached_count = 0
    failed_count = 0
    for qa in qas:
        qa_id = str(qa.get("id"))
        if qa_id in existing_results:
            cached_count += 1
        elif qa_id in failed_results:
            failed_count += 1
    new_count = len(qas) - cached_count - failed_count
    rerun_count = failed_count + new_count

    def get_llm(model_override: Optional[str] = None) -> EvaluatorLLM:
        if not hasattr(thread_local, "llm_by_model"):
            thread_local.llm_by_model = {}
        resolved_model = model_override or str(config.get("model") or "")
        cache_key = f"{provider}:{resolved_model}"
        if cache_key not in thread_local.llm_by_model:
            llm_config = dict(config)
            if model_override:
                llm_config["model"] = model_override
            thread_local.llm_by_model[cache_key] = EvaluatorLLM(provider, llm_config)
        return thread_local.llm_by_model[cache_key]

    def process_single(qa: Dict[str, Any]) -> Dict[str, Any]:
        qa_id = str(qa.get("id"))

        if qa_id in existing_results:
            return existing_results[qa_id]

        question = str(qa.get("question", ""))
        ground_truth = str(qa.get("answer", ""))
        prediction = predictions.get(qa_id, "")
        qtype = qtype_by_id.get(qa_id, QTYPE_OPEN)

        retry_count = 0
        last_error = None
        max_retries_effective = max_retries
        current_model = str(config.get("model") or "")
        fallback_model = ""
        if provider == "openai":
            fallback_model = str(config.get("fallback_model") or "").strip()
            if fallback_model == current_model:
                fallback_model = ""
        fallback_after_retries = max(
            0, int(config.get("fallback_after_retries", 0) or 0)
        )
        fallback_used = False
        fallback_trigger = ""

        def is_rate_limit_error(exc: Exception) -> bool:
            if isinstance(exc, requests.HTTPError):
                response = getattr(exc, "response", None)
                if response is not None and response.status_code == 429:
                    return True
            return "429" in str(exc)

        def maybe_switch_to_fallback(reason: str, *, forced: bool = False) -> bool:
            nonlocal current_model, fallback_used, fallback_trigger
            if not fallback_model or fallback_used or current_model == fallback_model:
                return False
            if not forced and retry_count < fallback_after_retries:
                return False
            fallback_used = True
            fallback_trigger = reason
            current_model = fallback_model
            print(
                f"\nSwitching judge model to {fallback_model} for QA {qa_id} after "
                f"{retry_count} failure(s): {reason}"
            )
            return True

        while retry_count < max_retries_effective:
            try:
                prompt = build_judge_prompt(question, ground_truth, prediction)
                response_text = get_llm(current_model).chat(
                    [{"role": "user", "content": prompt}]
                )
                parsed = parse_judge_response(response_text)
                accuracy_value = str(parsed.get("accuracy", "false")).lower() == "true"
                explanation = parsed.get("explanation", "")

                result = {
                    "id": qa_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "qtype": qtype,
                    "accuracy": accuracy_value,
                    "explanation": explanation,
                    "raw_response": response_text,
                    "judge_model": current_model,
                }

                if parsed.get("parse_fallback"):
                    result["parse_fallback"] = True

                if retry_count > 0:
                    result["retry_count"] = retry_count
                if fallback_used:
                    result["fallback_model_used"] = True
                    result["fallback_model"] = fallback_model
                    result["fallback_trigger"] = fallback_trigger

                if request_delay > 0:
                    time.sleep(request_delay)

                if output_path:
                    with write_lock:
                        write_incremental_result(output_path, result, existing_results)

                return result

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                rate_limited = is_rate_limit_error(e)
                non_retryable = isinstance(e, JudgeResponseError) and not e.retryable

                if rate_limited and max_retries_effective < 10:
                    max_retries_effective = 10

                if maybe_switch_to_fallback(last_error, forced=non_retryable):
                    continue

                if non_retryable:
                    print(
                        f"\nFailed QA {qa_id} with non-retriable judge error: {last_error}"
                    )
                    break

                if retry_count < max_retries_effective:
                    backoff_time = request_delay * (2 ** (retry_count - 1))
                    print(
                        f"\nRetry {retry_count}/{max_retries_effective} for QA {qa_id} after error: {last_error}"
                    )
                    print(f"Waiting {backoff_time:.1f}s before retry...")
                    time.sleep(backoff_time)
                else:
                    print(
                        f"\nFailed QA {qa_id} after {max_retries_effective} retries: {last_error}"
                    )

        result = {
            "id": qa_id,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "accuracy": False,
            "explanation": f"Failed after {retry_count} retries: {last_error}",
            "raw_response": f"ERROR after {retry_count} retries: {last_error}",
            "error": last_error,
            "retry_count": retry_count,
            "failed": True,
            "judge_model": current_model,
        }
        if fallback_used:
            result["fallback_model_used"] = True
            result["fallback_model"] = fallback_model
            result["fallback_trigger"] = fallback_trigger

        if output_path:
            with write_lock:
                write_incremental_result(output_path, result, existing_results)

        return result

    indexed_qas = list(enumerate(qas))
    cached_items: List[Tuple[int, Dict[str, Any]]] = []
    pending_items: List[Tuple[int, Dict[str, Any]]] = []

    for idx, qa in indexed_qas:
        qa_id = str(qa.get("id"))
        if qa_id in existing_results:
            cached_items.append((idx, existing_results[qa_id]))
        else:
            pending_items.append((idx, qa))

    skipped = len(cached_items)
    total = len(qas)
    model_name = str(config.get("model") or "unknown")
    desc = (
        f"LLM Judge {model_name}"
        f" (cached {cached_count}, failed {failed_count}, new {new_count}, rerun {rerun_count})"
    )

    rows: List[Dict[str, Any]] = []
    ordered_results: List[Tuple[int, Dict[str, Any]]] = list(cached_items)

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_single, qa): idx for idx, qa in pending_items
            }

            for future in tqdm(
                as_completed(future_map),
                total=total,
                initial=skipped,
                desc=desc,
                unit="qa",
            ):
                idx = future_map[future]
                result = future.result()
                ordered_results.append((idx, result))
    else:
        for idx, qa in tqdm(
            pending_items,
            total=total,
            initial=skipped,
            desc=desc,
            unit="qa",
        ):
            ordered_results.append((idx, process_single(qa)))

    ordered_results.sort(key=lambda item: item[0])
    rows.extend([item for _, item in ordered_results])

    for row in rows:
        if "qtype" not in row:
            row["qtype"] = qtype_by_id.get(str(row.get("id")), QTYPE_OPEN)

    correct = sum(1 for row in rows if row.get("accuracy"))
    summary = {
        "count": len(qas),
        "correct": correct,
        "accuracy": correct / len(qas) if qas else 0.0,
        "by_qtype": summarize_atm(rows)["by_qtype"],
    }
    if output_path and finalize_output:
        write_json_list(output_path, rows)
    return rows, summary


def build_prediction_map(predictions: List[Dict[str, Any]]) -> Dict[str, str]:
    result = {}
    for item in predictions:
        qa_id = str(item.get("id"))
        result[qa_id] = str(item.get("answer", ""))
    return result


def normalize_qtype_value(raw_value: Optional[str], answer: str) -> str:
    if raw_value:
        value = str(raw_value).strip().lower()
        if value in {"number", "num", "numeric"}:
            return QTYPE_NUMBER
        if value in {"list_recall", "list-recall", "list", "listrecall"}:
            return QTYPE_LIST
        if value in {
            "open_end",
            "open-end",
            "open",
            "openended",
            "open-ended",
            "open_ended",
        }:
            return QTYPE_OPEN
    return detect_qtype(answer)


def merge_atm_row(
    base: Dict[str, Any],
    qa: Dict[str, Any],
    prediction: str,
    qtype: str,
    metric: str,
) -> Dict[str, Any]:
    merged = dict(base)
    merged.setdefault("id", str(qa.get("id")))
    merged.setdefault("question", qa.get("question"))
    merged.setdefault("ground_truth", qa.get("answer"))
    merged.setdefault("prediction", prediction)
    merged["qtype"] = qtype
    merged["metric"] = metric
    return merged


def _coerce_accuracy(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def summarize_atm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        QTYPE_NUMBER: {"count": 0, "correct": 0},
        QTYPE_LIST: {"count": 0, "correct": 0},
        QTYPE_OPEN: {"count": 0, "correct": 0},
    }
    total_score = 0.0
    for row in rows:
        qtype = str(row.get("qtype") or QTYPE_OPEN).lower()
        if qtype not in totals:
            qtype = QTYPE_OPEN
        totals[qtype]["count"] += 1
        score = _coerce_accuracy(row.get("accuracy"))
        totals[qtype]["correct"] += score
        total_score += score

    summary = {
        "count": len(rows),
        "correct": total_score,
        "accuracy": total_score / len(rows) if rows else 0.0,
        "by_qtype": {},
    }
    for qtype, stats in totals.items():
        count = stats["count"]
        correct = stats["correct"]
        summary["by_qtype"][qtype] = {
            "count": count,
            "correct": correct,
            "accuracy": correct / count if count else 0.0,
        }
    return summary


def run_atm(
    qas: List[Dict[str, Any]],
    predictions: Dict[str, str],
    provider: str,
    config: Dict[str, Any],
    max_workers: int,
    request_delay: float,
    output_path: Optional[Path],
    llm_cache_path: Optional[Path],
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    existing_atm_all = load_existing_results(output_path) if output_path else {}
    llm_cache_all = load_existing_results(llm_cache_path) if llm_cache_path else {}

    existing_atm: Dict[str, Dict[str, Any]] = {}
    failed_atm: Dict[str, Dict[str, Any]] = {}
    for qa_id, result in existing_atm_all.items():
        if result.get("failed"):
            failed_atm[qa_id] = result
        else:
            existing_atm[qa_id] = result

    llm_cache: Dict[str, Dict[str, Any]] = {}
    failed_llm: Dict[str, Dict[str, Any]] = {}
    for qa_id, result in llm_cache_all.items():
        if result.get("failed"):
            failed_llm[qa_id] = result
        else:
            llm_cache[qa_id] = result

    qtype_by_id: Dict[str, str] = {}
    cached_atm = 0
    failed_atm_count = 0
    det_count = 0
    llm_cached = 0
    llm_failed = 0
    llm_new = 0

    for qa in qas:
        qa_id = str(qa.get("id"))
        answer = str(qa.get("answer", ""))
        qtype = normalize_qtype_value(qa.get("qtype"), answer)
        qtype_by_id[qa_id] = qtype

        # Deterministic ATM components (number EM + list Jaccard) are cheap and
        # can change when we update normalization/scoring logic. Recompute them
        # by default even if an earlier ATM checkpoint exists.
        if qtype in {QTYPE_NUMBER, QTYPE_LIST}:
            existing_atm.pop(qa_id, None)
            failed_atm.pop(qa_id, None)

        cached_row = existing_atm.get(qa_id)
        if cached_row:
            cached_metric = str(cached_row.get("metric") or "").lower()
            if qtype == QTYPE_LIST and cached_metric != "jaccard":
                existing_atm.pop(qa_id, None)
                cached_row = None
        if cached_row:
            cached_atm += 1
            continue
        if qa_id in failed_atm:
            failed_atm_count += 1

        if qtype in {QTYPE_NUMBER, QTYPE_LIST}:
            det_count += 1
            continue
        if qa_id in llm_cache:
            llm_cached += 1
        elif qa_id in failed_llm:
            llm_failed += 1
        else:
            llm_new += 1

    llm_rerun = llm_failed + llm_new
    desc = (
        "ATM"
        f" (cached {cached_atm}, failed {failed_atm_count}, det {det_count},"
        f" llm cached {llm_cached}, llm failed {llm_failed}, llm new {llm_new},"
        f" rerun {llm_rerun})"
    )

    rows: List[Optional[Dict[str, Any]]] = [None] * len(qas)
    pending_open: List[Tuple[int, Dict[str, Any]]] = []

    for idx, qa in tqdm(enumerate(qas), total=len(qas), desc=desc, unit="qa"):
        qa_id = str(qa.get("id"))
        prediction = predictions.get(qa_id, "")
        answer = str(qa.get("answer", ""))
        qtype = qtype_by_id.get(qa_id, QTYPE_OPEN)

        if qa_id in existing_atm:
            cached = existing_atm[qa_id]
            cached_metric = str(cached.get("metric") or "")
            if not cached_metric:
                if qtype == QTYPE_OPEN:
                    cached_metric = "llm"
                elif qtype == QTYPE_LIST:
                    cached_metric = "jaccard"
                else:
                    cached_metric = "em"
            cached_row = merge_atm_row(
                cached, qa, prediction, qtype, cached_metric
            )
            rows[idx] = cached_row
            continue

        if qtype == QTYPE_NUMBER:
            is_correct, normalized_prediction = _deterministic_accuracy_core(
                answer, prediction, question=str(qa.get("question", ""))
            )
            result = {
                "id": qa_id,
                "question": qa.get("question"),
                "ground_truth": answer,
                "prediction": prediction,
                "normalized_prediction": normalized_prediction,
                "qtype": qtype,
                "metric": "em",
                "accuracy": is_correct,
            }
            rows[idx] = result
            if output_path:
                write_incremental_result(output_path, result, existing_atm)
            continue
        # QTYPE_LIST uses Jaccard similarity instead of EM: list answers
        # require partial-match scoring since order is irrelevant and partial
        # recall matters (unlike QTYPE_NUMBER which uses strict EM).
        if qtype == QTYPE_LIST:
            score, pred_items_list = _list_jaccard_core(answer, prediction)
            result = {
                "id": qa_id,
                "question": qa.get("question"),
                "ground_truth": answer,
                "prediction": prediction,
                "normalized_prediction": ", ".join(pred_items_list),
                "qtype": qtype,
                "metric": "jaccard",
                "accuracy": score,
            }
            rows[idx] = result
            if output_path:
                write_incremental_result(output_path, result, existing_atm)
            continue

        if qa_id in llm_cache:
            cached = merge_atm_row(llm_cache[qa_id], qa, prediction, qtype, "llm")
            rows[idx] = cached
            if output_path:
                write_incremental_result(output_path, cached, existing_atm)
        else:
            pending_open.append((idx, qa))

    if pending_open:
        pending_qas = [qa for _, qa in pending_open]
        llm_rows, _ = run_llm_judge(
            pending_qas,
            predictions,
            provider=provider,
            config=config,
            max_workers=max_workers,
            request_delay=request_delay,
            output_path=llm_cache_path,
            max_retries=max_retries,
        )
        for (idx, qa), llm_row in zip(pending_open, llm_rows):
            prediction = predictions.get(str(qa.get("id")), "")
            result = merge_atm_row(llm_row, qa, prediction, QTYPE_OPEN, "llm")
            rows[idx] = result
            if output_path:
                write_incremental_result(output_path, result, existing_atm)

    final_rows = [row for row in rows if row is not None]
    summary = summarize_atm(final_rows)
    return final_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA evaluator")
    parser.add_argument(
        "--ground-truth",
        default=EVALUATOR_CONFIG["ground_truth_path"],
        help="Ground truth QA JSON file",
    )
    parser.add_argument(
        "--predictions",
        default=EVALUATOR_CONFIG["predictions_path"],
        help="Predictions JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default=EVALUATOR_CONFIG["output_dir"],
        help="Output directory for evaluator logs",
    )
    parser.add_argument(
        "--judge-provider",
        default=EVALUATOR_CONFIG["llm_judge"].get("provider", "openai"),
        choices=["openai", "vllm"],
        help="Provider for LLM judge",
    )
    parser.add_argument(
        "--judge-model",
        default=EVALUATOR_CONFIG["llm_judge"].get("model"),
        help="Model name for LLM judge",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Temperature for LLM judge (ignored for gpt-5/o-series)",
    )
    parser.add_argument(
        "--judge-endpoint",
        default=None,
        help="Override vLLM endpoint for LLM judge",
    )
    parser.add_argument(
        "--judge-thinking",
        default=None,
        choices=["enabled", "disabled"],
        help="Control thinking mode for judge (GLM-4.7 and similar models, default: disabled for vllm)",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        default=None,
        help="Reasoning effort for OpenAI judge models (e.g., none, minimal, low, medium, high)",
    )
    parser.add_argument(
        "--judge-fallback-model",
        default=EVALUATOR_CONFIG["llm_judge"].get("fallback_model", "gpt-5"),
        help="Fallback OpenAI judge model to use after repeated failures (default: gpt-5)",
    )
    parser.add_argument(
        "--judge-fallback-after-retries",
        type=int,
        default=EVALUATOR_CONFIG["llm_judge"].get("fallback_after_retries", 3),
        help="Switch to the fallback judge model after this many failed attempts on the primary model (default: 3)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=10.0,
        help="Delay in seconds between LLM judge requests (default: 10.0)",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=10,
        help="Max retries per LLM judge request (default: 10)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["em", "llm", "atm"],
        default=["em", "llm", "atm"],
        help="Metrics to run (default: em llm atm)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max workers for LLM judge concurrency",
    )
    return parser.parse_args()


def build_judge_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.judge_provider == "vllm":
        judge_config = dict(EVALUATOR_CONFIG.get("vllm_text", {}))
    else:
        judge_config = dict(EVALUATOR_CONFIG["llm_judge"])

    judge_config["provider"] = args.judge_provider
    judge_config["model"] = args.judge_model
    if args.judge_endpoint:
        judge_config["endpoint"] = args.judge_endpoint
    if args.judge_temperature is not None:
        judge_config["temperature"] = args.judge_temperature
    elif args.judge_provider == "vllm":
        judge_config["temperature"] = judge_config.get("temperature", 0.0)

    if args.judge_thinking:
        judge_config["thinking_mode"] = args.judge_thinking
    elif args.judge_provider == "vllm" and "thinking_mode" not in judge_config:
        judge_config["thinking_mode"] = "disabled"

    if args.judge_reasoning_effort:
        judge_config["reasoning_effort"] = args.judge_reasoning_effort
    judge_config["fallback_model"] = str(args.judge_fallback_model or "").strip()
    judge_config["fallback_after_retries"] = max(
        0, int(args.judge_fallback_after_retries)
    )

    return judge_config


def seed_llm_from_atm(atm_path: Path) -> Dict[str, Dict[str, Any]]:
    if not atm_path.exists():
        return {}
    cached = load_existing_results(atm_path)
    seed = {}
    for qa_id, row in cached.items():
        qtype = str(row.get("qtype", "")).lower()
        metric = str(row.get("metric", "")).lower()
        if qtype == QTYPE_OPEN or metric == "llm":
            seed[qa_id] = row
    return seed


def main() -> None:
    args = parse_args()
    ground_truth_path = Path(args.ground_truth)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    qa_data = load_json(ground_truth_path)
    qas = load_qa_list(qa_data)
    predictions_data = load_jsonl(predictions_path)
    predictions = build_prediction_map(predictions_data)

    if not args.metrics:
        selected = {"em", "llm", "atm"}
    else:
        selected = {metric.lower() for metric in args.metrics}

    if "em" in selected:
        deterministic_rows, deterministic_summary = run_deterministic(qas, predictions)
        write_json_list(output_dir / "deterministic_accuracy.json", deterministic_rows)
        write_json(
            output_dir / "deterministic_accuracy_summary.json", deterministic_summary
        )

    needs_llm = bool({"llm", "atm"} & selected)
    if needs_llm:
        judge_config = build_judge_config(args)
        model_tag = safe_filename(str(args.judge_model))
        judge_json = output_dir / f"llm_judge_{model_tag}.json"
        judge_summary_path = output_dir / f"llm_judge_{model_tag}_summary.json"
        atm_json = output_dir / f"atm_{model_tag}.json"
        atm_summary_path = output_dir / f"atm_{model_tag}_summary.json"

    if "atm" in selected:
        atm_rows, atm_summary = run_atm(
            qas,
            predictions,
            provider=args.judge_provider,
            config=judge_config,
            max_workers=args.max_workers,
            request_delay=args.request_delay,
            output_path=atm_json,
            llm_cache_path=judge_json,
            max_retries=args.judge_max_retries,
        )
        write_json_list(atm_json, atm_rows)
        write_json(atm_summary_path, atm_summary)

    if "llm" in selected:
        # Seed LLM judge with ATM results only when ATM was also computed.
        seed_results = seed_llm_from_atm(atm_json) if "atm" in selected else {}
        judge_rows, judge_summary = run_llm_judge(
            qas,
            predictions,
            provider=args.judge_provider,
            config=judge_config,
            max_workers=args.max_workers,
            request_delay=args.request_delay,
            output_path=judge_json,
            max_retries=args.judge_max_retries,
            seed_results=seed_results,
        )

        write_json(judge_summary_path, judge_summary)


if __name__ == "__main__":
    main()
