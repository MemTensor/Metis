#!/usr/bin/env python3
"""Normalization utilities for QA evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Iterable, List, Optional, Tuple

from memqa.utils.evaluator.config import ABSTENTION_PHRASES

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_PATTERN = "|".join(MONTHS.keys())

RE_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
RE_COMPACT_DATE = re.compile(
    r"\b((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b"
)
RE_MONTH_DAY = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_DAY_MONTH = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_MONTH_RANGE = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:-|to|–)\s*(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
RE_TIME_RANGE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:-|to|–|and)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
    re.IGNORECASE,
)
RE_TIME_AMPM = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
RE_TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

RE_CURRENCY_SYMBOL = re.compile(r"[£$€]")
RE_CURRENCY_CODE = re.compile(r"\b(USD|GBP|EUR|AUD|JOD)\b", re.IGNORECASE)
RE_CURRENCY_WORDS = re.compile(
    r"\b(dollars?|usd|pounds?|gbp|euros?|eur)\b", re.IGNORECASE
)
RE_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
RE_CODE = re.compile(r"\b[A-Z0-9]{4,}\b", re.IGNORECASE)
RE_TIME_AMPM_DOTS = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.m\.|p\.m\.)\b", re.IGNORECASE
)
RE_CURRENCY_AMOUNT_PREFIX = re.compile(
    r"(?P<symbol>[£$€])\s*(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
)
RE_CURRENCY_AMOUNT_SUFFIX = re.compile(
    r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?P<code>USD|GBP|EUR|AUD|JOD|dollars?|pounds?|euros?)\b",
    re.IGNORECASE,
)

LIST_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\band\b|/)\s*", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "on",
    "for",
    "to",
    "and",
    "or",
    "by",
    "from",
    "with",
    "into",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
}

ADDRESS_KEYWORDS = {
    "street",
    "st",
    "road",
    "rd",
    "lane",
    "ln",
    "avenue",
    "ave",
    "drive",
    "dr",
    "place",
    "pl",
    "square",
    "sq",
    "close",
    "court",
    "ct",
    "way",
    "high",
    "highstreet",
    "boulevard",
    "blvd",
    "city",
    "town",
    "village",
    "county",
    "country",
    "oxford",
    "cambridge",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

DAYS_OF_WEEK = {
    "monday",
    "mon",
    "tuesday",
    "tue",
    "tues",
    "wednesday",
    "wed",
    "thursday",
    "thu",
    "thurs",
    "friday",
    "fri",
    "saturday",
    "sat",
    "sunday",
    "sun",
}

DAY_PATTERN = re.compile(r"\b(" + "|".join(DAYS_OF_WEEK) + r")\b", re.IGNORECASE)


@dataclass(frozen=True)
class DateToken:
    value: str


@dataclass(frozen=True)
class TimeToken:
    value: str


def aggressive_preprocess(text: str) -> str:
    """Aggressive preprocessing to normalize semantically equivalent variations."""
    cleaned = text

    # Strip trailing periods from entire text
    cleaned = cleaned.rstrip(".")

    # Normalize comma separators in numbers
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

    # Strip trailing unit words that add no info ("spots", "hours", etc.)
    cleaned = re.sub(
        r"\b(\d+)\s+(spots?|hours?|sessions?|days?)\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", cleaned)
    cleaned = re.sub(r"\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip commas from numbers
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

    cleaned = cleaned.strip("\t\n\r \"'`.,;:!?()[]{}")
    return cleaned


def strip_parenthetical_details(text: str) -> str:
    """Pattern 1: Remove parenthetical clarifications and extra details.

    Examples:
    - "£12.85 (with 16-25 RAILCARD discount)" -> "£12.85"
    - "Tomorrow at 4:00 PM (March 1, 2022)" -> "Tomorrow at 4:00 PM"
    """
    cleaned = re.sub(r"\([^)]+\)", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_currency_breakdowns(text: str) -> str:
    """Pattern 2: Remove detailed currency breakdowns after amounts.

    Examples:
    - "£10 per 60-minute session, totaling £30 for three sessions" -> "£10 per session"
    - "The total cost was £190.50, comprising £110..." -> "£190.50"
    """
    cleaned = text
    # Remove "totaling X for Y sessions" patterns
    cleaned = re.sub(
        r",?\s*totaling\s+[£$€]\d+(?:\.\d+)?\s+for\s+\w+\s+sessions?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove "comprising X..." patterns
    cleaned = re.sub(
        r",?\s*comprising\s+.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove "per X-minute session" -> "per session"
    cleaned = re.sub(
        r"per\s+\d+[- ]minute\s+session",
        "per session",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_between_to_range(text: str) -> str:
    """Pattern 6: Normalize 'between X and Y' to 'X to Y'.

    Examples:
    - "Between April 20 and April 26" -> "April 20 to April 26"
    """
    # Handle "between X and Y" -> "X to Y"
    cleaned = re.sub(
        r"\bbetween\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+and\s+",
        " to ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def strip_leading_articles(text: str) -> str:
    """Pattern 7: Strip leading articles (A, An, The).

    Examples:
    - "A Star Wars-themed event" -> "Star Wars-themed event"
    """
    cleaned = re.sub(r"^\s*(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return cleaned


def strip_context_phrases(text: str) -> str:
    """Remove common contextual phrases that don't change the core answer."""
    cleaned = text.lower()

    context_patterns = [
        r"\bit happened on\b",
        r"\bstarting at\b",
        r"\bfrom\s+\d",
        r"\bonwards\b",
        r"\btotaling\b",
        r"\bcomprising\b",
        r"\bwith pre-drinks at\b",
        r"\bwith pre-\w+\b",
        r"\band dinner at\b",
        r"\bdress code is\b",
        r"\bthe total cost was\b",
        r"\bcode:\s*",
        r"\bexpires?\s*",
    ]

    for pattern in context_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = DAY_PATTERN.sub(" ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned




def is_abstention(text: str) -> bool:
    normalized = normalize_text(text)
    for phrase in ABSTENTION_PHRASES:
        if phrase in normalized:
            return True
    return False


def _month_to_number(name: str) -> Optional[int]:
    return MONTHS.get(name.lower())


def _format_date(year: Optional[int], month: int, day: int) -> str:
    if year:
        return f"{year:04d}{month:02d}{day:02d}"
    return f"{month:02d}{day:02d}"


def _format_range(year: Optional[int], month: int, start_day: int, end_day: int) -> str:
    if year:
        start = f"{year:04d}{month:02d}{start_day:02d}"
        end = f"{year:04d}{month:02d}{end_day:02d}"
    else:
        start = f"{month:02d}{start_day:02d}"
        end = f"{month:02d}{end_day:02d}"
    return f"{start}-{end}"


def extract_dates(text: str) -> List[DateToken]:
    tokens: List[DateToken] = []

    for match in RE_ISO_DATE.finditer(text):
        year, month, day = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_COMPACT_DATE.finditer(text):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_MONTH_RANGE.finditer(text):
        month = _month_to_number(match.group(1))
        if not month:
            continue
        start_day = int(match.group(2))
        end_day = int(match.group(3))
        year = int(match.group(4)) if match.group(4) else None
        tokens.append(DateToken(_format_range(year, month, start_day, end_day)))

    text_without_ranges = RE_MONTH_RANGE.sub(" ", text)

    for match in RE_MONTH_DAY.finditer(text_without_ranges):
        month = _month_to_number(match.group(1))
        if not month:
            continue
        day = int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        tokens.append(DateToken(_format_date(year, month, day)))

    for match in RE_DAY_MONTH.finditer(text_without_ranges):
        day = int(match.group(1))
        month = _month_to_number(match.group(2))
        if not month:
            continue
        year = int(match.group(3)) if match.group(3) else None
        tokens.append(DateToken(_format_date(year, month, day)))

    return dedupe_tokens(tokens)


def _normalize_time(hour: int, minute: int, ampm: Optional[str]) -> str:
    if ampm:
        ampm = ampm.lower()
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour += 12
    return f"{hour:02d}:{minute:02d}"


def extract_times(text: str) -> List[TimeToken]:
    tokens: List[TimeToken] = []

    normalized = text.lower()
    normalized = normalized.replace("noon", "12:00")
    normalized = normalized.replace("midday", "12:00")
    normalized = normalized.replace("midnight", "00:00")

    normalized = normalized.replace("a.m.", "am").replace("p.m.", "pm")
    normalized = normalized.replace("a. m.", "am").replace("p. m.", "pm")

    normalized = RE_TIME_AMPM_DOTS.sub(
        lambda m: f"{m.group(1)}:{m.group(2) or '00'}{m.group(3)[0]}m", normalized
    )

    for match in RE_TIME_RANGE.finditer(normalized):
        start_hour = int(match.group(1))
        start_min = int(match.group(2) or 0)
        start_ampm = match.group(3)
        end_hour = int(match.group(4))
        end_min = int(match.group(5) or 0)
        end_ampm = match.group(6)
        if start_ampm is None and end_ampm is not None:
            start_ampm = end_ampm
        start = _normalize_time(start_hour, start_min, start_ampm)
        end = _normalize_time(end_hour, end_min, end_ampm)
        tokens.append(TimeToken(f"{start}-{end}"))

    text_without_ranges = RE_TIME_RANGE.sub(" ", normalized)

    for match in RE_TIME_24H.finditer(text_without_ranges):
        hour = int(match.group(1))
        minute = int(match.group(2))
        tokens.append(TimeToken(_normalize_time(hour, minute, None)))

    for match in RE_TIME_AMPM.finditer(text_without_ranges):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3)
        tokens.append(TimeToken(_normalize_time(hour, minute, ampm)))

    return dedupe_tokens(tokens)


def remove_date_time_text(text: str) -> str:
    cleaned = RE_ISO_DATE.sub(" ", text)
    cleaned = RE_COMPACT_DATE.sub(" ", cleaned)
    cleaned = RE_MONTH_RANGE.sub(" ", cleaned)
    cleaned = RE_MONTH_DAY.sub(" ", cleaned)
    cleaned = RE_DAY_MONTH.sub(" ", cleaned)
    cleaned = RE_TIME_RANGE.sub(" ", cleaned)
    cleaned = RE_TIME_24H.sub(" ", cleaned)
    cleaned = RE_TIME_AMPM.sub(" ", cleaned)
    cleaned = RE_TIME_AMPM_DOTS.sub(" ", cleaned)
    cleaned = (
        cleaned.replace("noon", " ").replace("midday", " ").replace("midnight", " ")
    )
    cleaned = cleaned.replace("a.m.", " ").replace("p.m.", " ")
    cleaned = cleaned.replace("a. m.", " ").replace("p. m.", " ")
    return cleaned


def _normalize_decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _replace_number_words(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(text)
    for token in tokens:
        lower = token.lower()
        if lower in NUMBER_WORDS:
            text = re.sub(rf"\b{re.escape(token)}\b", str(NUMBER_WORDS[lower]), text)
    return text


def extract_currency_amounts(text: str) -> List[Decimal]:
    amounts: List[Decimal] = []

    normalized = _replace_number_words(text)

    # Remove commas from numbers before extraction
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)

    for match in RE_CURRENCY_AMOUNT_PREFIX.finditer(normalized):
        number = _normalize_decimal(match.group("number"))
        if number is not None:
            amounts.append(number.normalize())

    for match in RE_CURRENCY_AMOUNT_SUFFIX.finditer(normalized):
        number = _normalize_decimal(match.group("number"))
        if number is not None:
            amounts.append(number.normalize())

    return dedupe_tokens(amounts)


def extract_numbers(text: str) -> Tuple[List[Decimal], List[str]]:
    values: List[Decimal] = []
    currencies: List[str] = []

    for match in RE_CURRENCY_CODE.finditer(text):
        currencies.append(match.group(1).upper())

    for match in RE_CURRENCY_WORDS.finditer(text):
        word = match.group(1).lower()
        if "usd" in word or "dollar" in word:
            currencies.append("USD")
        elif "gbp" in word or "pound" in word:
            currencies.append("GBP")
        elif "eur" in word or "euro" in word:
            currencies.append("EUR")

    symbol_map = {"£": "GBP", "$": "USD", "€": "EUR"}
    for symbol_match in RE_CURRENCY_SYMBOL.finditer(text):
        symbol = symbol_match.group(0)
        mapped = symbol_map.get(symbol)
        if mapped:
            currencies.append(mapped)

    normalized = _replace_number_words(text)
    for match in RE_NUMBER.finditer(normalized):
        number = _normalize_decimal(match.group(0))
        if number is not None:
            values.append(number.normalize())

    return values, currencies


def extract_codes(text: str) -> List[str]:
    tokens: List[str] = []
    for match in RE_CODE.finditer(text):
        value = match.group(0)
        # Skip patterns that look like dates with ordinal suffixes (e.g., "25TH", "1ST")
        if re.match(r"^\d{1,2}(ST|ND|RD|TH)$", value.upper()):
            continue
        if any(char.isalpha() for char in value) and any(
            char.isdigit() for char in value
        ):
            tokens.append(value.upper())
    return dedupe_tokens(tokens)


def normalize_currency_codes(currencies: List[str]) -> List[str]:
    normalized = set()
    for curr in currencies:
        curr_upper = curr.upper()
        if curr_upper in ["GBP", "POUNDS", "POUND", "£"]:
            normalized.add("GBP")
        elif curr_upper in ["USD", "DOLLARS", "DOLLAR", "$"]:
            normalized.add("USD")
        elif curr_upper in ["EUR", "EUROS", "EURO", "€"]:
            normalized.add("EUR")
        elif curr_upper in ["AUD"]:
            normalized.add("AUD")
        elif curr_upper in ["JOD"]:
            normalized.add("JOD")
        else:
            normalized.add(curr_upper)
    return sorted(list(normalized))


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def split_list_items(text: str) -> List[str]:
    items = [item.strip() for item in LIST_SPLIT_PATTERN.split(text) if item.strip()]
    return [normalize_text(item) for item in items if item]


def token_subset_match(ground_truth: str, prediction: str) -> bool:
    """Check if all non-stopword GT tokens appear in prediction.

    Returns True if GT is empty/all-stopwords (vacuous truth).
    """
    gt_tokens = [token for token in tokenize(ground_truth) if token not in STOPWORDS]
    pred_tokens = set(tokenize(prediction))

    # Empty GT (all stopwords removed) → vacuously true
    if not gt_tokens:
        return True

    if len(gt_tokens) == 1:
        return gt_tokens[0] in pred_tokens
    return all(token in pred_tokens for token in gt_tokens)


def location_token_match(ground_truth: str, prediction: str) -> bool:
    gt_tokens = tokenize(ground_truth)
    if not any(token in ADDRESS_KEYWORDS for token in gt_tokens):
        return False
    filtered = [
        token for token in gt_tokens if token not in STOPWORDS and not token.isdigit()
    ]
    pred_tokens = set(tokenize(prediction))
    if not filtered:
        return False
    return all(token in pred_tokens for token in filtered)


def extract_reference_date(text: str) -> Optional[date]:
    tokens = extract_dates(text)
    for token in tokens:
        value = token.value
        if len(value) == 8:
            year = int(value[0:4])
            month = int(value[4:6])
            day = int(value[6:8])
            return date(year, month, day)
    return None


def resolve_relative_dates(text: str, reference: Optional[date]) -> str:
    if not reference:
        return text
    mapping = {
        "today": reference,
        "tonight": reference,
        "tomorrow": reference + timedelta(days=1),
        "yesterday": reference - timedelta(days=1),
    }
    resolved = text
    for key, value in mapping.items():
        iso_value = value.strftime("%Y-%m-%d")
        month_day = value.strftime("%B %d")

        # Replace "Tomorrow" with both ISO date and readable format
        if key in resolved.lower():
            # Try to preserve capitalization
            for pattern in [key, key.capitalize(), key.upper()]:
                if pattern in resolved:
                    # Replace with month-day format for better matching
                    resolved = resolved.replace(pattern, month_day)
                    break

    # Remove parenthetical date explanations after relative dates
    resolved = re.sub(
        r"(today|tonight|tomorrow|yesterday)\s*\([^)]+\)",
        r"\1",
        resolved,
        flags=re.IGNORECASE,
    )

    return resolved


def semantic_units_match(ground_truth: str, prediction: str) -> bool:
    """Ultra-aggressive semantic matching: if all key units match, accept.

    Returns True if GT and PRED are semantically equivalent based on:
    - Dates match (ignoring ordinals, years)
    - Times match
    - Numbers/currencies match
    - Key tokens overlap substantially
    """
    gt_dates = extract_dates(ground_truth)
    pr_dates = extract_dates(prediction)
    gt_times = extract_times(ground_truth)
    pr_times = extract_times(prediction)
    gt_amounts = extract_currency_amounts(ground_truth)
    pr_amounts = extract_currency_amounts(prediction)

    # If GT has dates, PR must have matching dates
    if gt_dates:
        if not pr_dates:
            return False
        gt_date_vals = {d.value for d in gt_dates}
        pr_date_vals = {d.value for d in pr_dates}
        # Allow flexible date matching (e.g., "0725" matches "20220725")
        if not any(
            gt_d[-4:] == pr_d[-4:] for gt_d in gt_date_vals for pr_d in pr_date_vals
        ):
            return False

    # If GT has times, PR should have matching times (but not required)
    if gt_times and pr_times:
        gt_time_vals = {t.value for t in gt_times}
        pr_time_vals = {t.value for t in pr_times}
        if not gt_time_vals.intersection(pr_time_vals):
            return False

    # If GT has currency amounts, PR must have them
    if gt_amounts:
        if not pr_amounts:
            return False
        if set(gt_amounts) != set(pr_amounts):
            return False

    # Token overlap check
    gt_tokens = set(tokenize(ground_truth))
    pr_tokens = set(tokenize(prediction))

    # Remove stopwords
    gt_content = {t for t in gt_tokens if t not in STOPWORDS and len(t) > 2}
    pr_content = {t for t in pr_tokens if t not in STOPWORDS and len(t) > 2}

    if not gt_content:
        return True

    # If 80%+ of GT content tokens are in PR, it's a match
    overlap = len(gt_content.intersection(pr_content))
    if overlap / len(gt_content) >= 0.8:
        return True

    return False


def dedupe_tokens(tokens: Iterable) -> List:
    seen = set()
    result = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result
