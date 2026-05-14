from __future__ import annotations

import re


ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(-?\d+)", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
INT_RE = re.compile(r"(?<![\w.])-?\d+(?![\w.])")


def parse_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match:
        return normalize_int(match.group(1))
    ints = INT_RE.findall(text)
    if not ints:
        return None
    return normalize_int(ints[-1])


def normalize_int(value: str | int) -> str:
    return str(int(value))


def parse_confidence(text: str) -> float | None:
    match = CONF_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def has_required_format(text: str) -> bool:
    return ANSWER_RE.search(text) is not None and CONF_RE.search(text) is not None
