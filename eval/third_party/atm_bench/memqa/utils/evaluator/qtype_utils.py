#!/usr/bin/env python3
"""Utilities for classifying QA examples by answer type."""

from __future__ import annotations

import re
from typing import Iterable, List

from memqa.utils.evaluator.normalizer import (
    ADDRESS_KEYWORDS,
    DAY_PATTERN,
    STOPWORDS,
    aggressive_preprocess,
    extract_currency_amounts,
    extract_dates,
    extract_numbers,
    extract_times,
    is_abstention,
    normalize_between_to_range,
    remove_date_time_text,
    strip_currency_breakdowns,
    strip_parenthetical_details,
    tokenize,
)

QTYPE_NUMBER = "number"
QTYPE_LIST = "list_recall"
QTYPE_OPEN = "open_end"

LIST_SEPARATOR_PATTERN = re.compile(r"[,;\n/]+")
EVIDENCE_IMAGE_PATTERN = re.compile(r"\b\d{8}_\d{6}(?:_resized)?\b")
EVIDENCE_EMAIL_PATTERN = re.compile(r"\bemail\d{12}\b", re.IGNORECASE)
RE_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)
RE_US_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

VERB_TOKENS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
}

NUMBER_CONNECTORS = re.compile(
    r"\b(on|at|from|to|between|and|or|around|about|approx|approximately|before|after|"
    r"until|till|by|in|of|for)\b",
    re.IGNORECASE,
)

CURRENCY_WORDS = re.compile(
    r"\b(usd|gbp|eur|aud|jod|dollars?|pounds?|euros?)\b", re.IGNORECASE
)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_evidence_ids(text: str) -> List[str]:
    ids = []
    ids.extend(EVIDENCE_IMAGE_PATTERN.findall(text))
    ids.extend(EVIDENCE_EMAIL_PATTERN.findall(text))
    return ids


def _strip_item(item: str) -> str:
    stripped = item.strip().strip("\"'`")
    stripped = stripped.strip(" .,:;")
    return stripped.strip()


def _has_alpha(tokens: Iterable[str]) -> bool:
    return any(any(char.isalpha() for char in token) for token in tokens)


def _is_title_like(item: str) -> bool:
    cleaned = _strip_item(item)
    if not cleaned:
        return False
    tokens = tokenize(cleaned)
    if not tokens:
        return False
    if len(tokens) > 12:
        return False
    if not _has_alpha(tokens):
        return False
    if any(token in VERB_TOKENS for token in tokens):
        return False
    stopword_ratio = sum(1 for token in tokens if token in STOPWORDS) / len(tokens)
    if stopword_ratio > 0.6:
        return False
    return True


def _has_list_conjunction(text: str) -> bool:
    lowered = text.lower()
    return " and " in lowered or " or " in lowered or ";" in lowered or "/" in lowered


def _is_location_like(answer: str) -> bool:
    tokens = tokenize(answer)
    if any(token in ADDRESS_KEYWORDS for token in tokens):
        return True
    if RE_UK_POSTCODE.search(answer) or RE_US_ZIP.search(answer):
        return True
    if _has_list_conjunction(answer):
        return False
    segments = [seg.strip() for seg in answer.split(",") if seg.strip()]
    if len(segments) < 3:
        return False
    short_segments = 0
    for seg in segments:
        seg_tokens = tokenize(seg)
        if len(seg_tokens) <= 3:
            short_segments += 1
    return short_segments == len(segments)


def is_number_answer(answer: str) -> bool:
    if not answer or is_abstention(answer):
        return False

    cleaned = aggressive_preprocess(answer)
    cleaned = strip_parenthetical_details(cleaned)
    cleaned = strip_currency_breakdowns(cleaned)
    cleaned = normalize_between_to_range(cleaned)

    has_numeric = bool(extract_dates(cleaned) or extract_times(cleaned))
    numbers, currencies = extract_numbers(cleaned)
    currency_amounts = extract_currency_amounts(cleaned)
    has_numeric = has_numeric or bool(numbers or currencies or currency_amounts)
    if not has_numeric:
        return False

    remainder = remove_date_time_text(cleaned)
    remainder = DAY_PATTERN.sub(" ", remainder)
    remainder = NUMBER_CONNECTORS.sub(" ", remainder)
    remainder = CURRENCY_WORDS.sub(" ", remainder)
    remainder = re.sub(r"[£$€]", " ", remainder)
    remainder = re.sub(r"\d", " ", remainder)
    remainder = re.sub(r"[-–~/:,.;]", " ", remainder)
    remainder = _compact_whitespace(remainder)
    return not remainder


def _is_evidence_id_list(answer: str) -> bool:
    ids = _extract_evidence_ids(answer)
    if not ids:
        return False

    remainder = answer
    for item_id in ids:
        remainder = remainder.replace(item_id, " ")
    remainder = LIST_SEPARATOR_PATTERN.sub(" ", remainder)
    remainder = re.sub(r"[\s\-–~:.]+", " ", remainder).strip()
    if re.search(r"[A-Za-z0-9]", remainder):
        return False
    return True


def is_list_answer(answer: str) -> bool:
    if not answer:
        return False
    return _is_evidence_id_list(answer)


def detect_qtype(answer: str) -> str:
    if is_number_answer(answer):
        return QTYPE_NUMBER
    if is_list_answer(answer):
        return QTYPE_LIST
    return QTYPE_OPEN


__all__ = [
    "QTYPE_NUMBER",
    "QTYPE_LIST",
    "QTYPE_OPEN",
    "detect_qtype",
    "is_list_answer",
    "is_number_answer",
]
