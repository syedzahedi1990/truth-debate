from __future__ import annotations

import re


INT_VALUE = r"([+-]?\d[\d,]*)"
ANSWER_RE = re.compile(rf"(?<![A-Za-z_])ANSWER\s*[:=]\s*(?:<integer>\s*[:=]?\s*)?\**\s*{INT_VALUE}", re.IGNORECASE)
FINAL_RE = re.compile(
    rf"\bfinal\s+(?:answer|integer|result)\s*(?:(?:is|:|=)\s*)?(?:[:=]\s*)?(?:<integer>\s*[:=]?\s*)?\**\s*{INT_VALUE}",
    re.IGNORECASE,
)
INTEGER_TAG_RE = re.compile(rf"<integer>\s*[:=]\s*{INT_VALUE}", re.IGNORECASE)
BOXED_RE = re.compile(rf"\\boxed\s*\{{\s*{INT_VALUE}\s*\}}", re.IGNORECASE)
MARKER_ONLY_RE = re.compile(
    r"(?<![A-Za-z_])ANSWER\b|\bfinal\s+(?:answer|integer|result)\b|<integer>\s*[:=]?\s*$",
    re.IGNORECASE,
)
NEXT_LINE_VALUE_RE = re.compile(rf"^\s*(?:[`'\"*_]+\s*)?{INT_VALUE}(?:\s*[`'\"*_.!,;]*)?\s*$")
CONF_RE = re.compile(r"CONFIDENCE\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
INT_RE = re.compile(r"(?<![\w.])-?\d+(?![\w.])")


def parse_answer(text: str) -> str | None:
    """Parse a deliberately marked final answer.

    This parser is intentionally conservative. It avoids the old last-integer
    fallback because debate outputs often contain confidence values, numbered
    lists, and quoted peer answers that are not the model's final answer.
    """
    candidates: list[tuple[int, int, str]] = []
    lines = text.splitlines()
    for line_idx, line in enumerate(lines):
        clean = line.strip()
        if not clean or _ignore_answer_line(clean):
            continue
        for priority, regex in enumerate((ANSWER_RE, FINAL_RE, BOXED_RE, INTEGER_TAG_RE)):
            for match in regex.finditer(clean):
                value = normalize_int(match.group(1))
                candidates.append((priority, line_idx, value))
        if MARKER_ONLY_RE.search(clean):
            next_value = _parse_next_line_value(lines[line_idx + 1 : line_idx + 4])
            if next_value is not None:
                candidates.append((1, line_idx, next_value))

    if not candidates:
        return None
    best_priority = min(priority for priority, _, _ in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_priority]
    return sorted(best, key=lambda item: item[1])[-1][2]


def parse_answer_legacy(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match:
        return normalize_int(match.group(1))
    legacy_answer = re.search(r"ANSWER\s*[:=]\s*(-?\d+)", text, re.IGNORECASE)
    if legacy_answer:
        return normalize_int(legacy_answer.group(1))
    match = ANSWER_RE.search(text)
    if match:
        return normalize_int(match.group(1))
    ints = INT_RE.findall(text)
    if not ints:
        return None
    return normalize_int(ints[-1])


def normalize_int(value: str | int) -> str:
    return str(int(str(value).replace(",", "")))


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


def _ignore_answer_line(line: str) -> bool:
    low = line.lower()
    if "private_answer" in low or "private answer" in low:
        return True
    if re.search(r"\bpeer\s*\d*\s*(?:'s)?\s+answer\b", low):
        return True
    if low.startswith("peer "):
        return True
    if "confidence" in low and not re.search(r"answer|final|result|<integer>|boxed", low):
        return True
    return False


def _parse_next_line_value(lines: list[str]) -> str | None:
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if "confidence" in clean.lower():
            return None
        match = NEXT_LINE_VALUE_RE.search(clean)
        return normalize_int(match.group(1)) if match else None
    return None
