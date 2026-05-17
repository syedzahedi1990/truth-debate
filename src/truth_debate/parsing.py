from __future__ import annotations

import json
import re


INT_VALUE = r"([+-]?\d[\d,]*)"
JSON_ANSWER_RE = re.compile(rf'"answer"\s*:\s*"?{INT_VALUE}"?', re.IGNORECASE)
JSON_CONF_RE = re.compile(r'"confidence"\s*:\s*"?(\d+(?:\.\d+)?)"?', re.IGNORECASE)
GSM8K_ANSWER_RE = re.compile(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)")
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
NUMERIC_RE = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?!\w)")


def parse_answer(text: str) -> str | None:
    """Parse a deliberately marked final answer.

    This parser is intentionally conservative. It avoids the old last-integer
    fallback because debate outputs often contain confidence values, numbered
    lists, and quoted peer answers that are not the model's final answer.
    """
    json_answer = _parse_json_answer(text)
    if json_answer is not None:
        return json_answer

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


def parse_standard_numeric_answer(text: str) -> str | None:
    """Parse a benchmark-style numeric answer without changing strict metrics."""
    strict = parse_answer(text)
    if strict is not None:
        return strict

    marker = GSM8K_ANSWER_RE.search(text)
    if marker:
        return normalize_int(marker.group(1))

    candidates: list[str] = []
    for match in NUMERIC_RE.finditer(text):
        if _numeric_match_is_confidence(text, match):
            continue
        try:
            candidates.append(normalize_int(match.group(0)))
        except ValueError:
            continue
    if not candidates:
        return None
    return candidates[-1]


def normalize_int(value: str | int) -> str:
    if isinstance(value, float):
        return str(int(value))
    clean = str(value).replace(",", "").strip()
    if "." in clean:
        return str(int(float(clean)))
    return str(int(clean))


def parse_confidence(text: str) -> float | None:
    json_conf = _parse_json_confidence(text)
    if json_conf is not None:
        return json_conf
    match = CONF_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def has_required_format(text: str) -> bool:
    if _parse_json_answer(text) is not None and _parse_json_confidence(text) is not None:
        return True
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


def _parse_json_answer(text: str) -> str | None:
    parsed_answers: list[str] = []
    for obj in _json_objects(text):
        answer = obj.get("answer")
        if isinstance(answer, (int, float, str)):
            try:
                parsed_answers.append(normalize_int(answer))
            except ValueError:
                pass
    if parsed_answers:
        return parsed_answers[-1]
    matches = JSON_ANSWER_RE.findall(text)
    if matches:
        return normalize_int(matches[-1])
    return None


def _parse_json_confidence(text: str) -> float | None:
    parsed_confidences: list[float] = []
    for obj in _json_objects(text):
        conf = obj.get("confidence")
        if isinstance(conf, (int, float, str)):
            try:
                value = float(conf)
            except ValueError:
                continue
            if value > 1:
                value /= 100.0
            parsed_confidences.append(max(0.0, min(1.0, value)))
    if parsed_confidences:
        return parsed_confidences[-1]
    matches = JSON_CONF_RE.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    if value > 1:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _numeric_match_is_confidence(text: str, match: re.Match[str]) -> bool:
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    if end == -1:
        end = len(text)
    line = text[start:end].lower()
    if "confidence" in line and not re.search(r"answer|final|result|####", line):
        return True
    tail = text[match.end() : match.end() + 1]
    return tail == "%"


def _json_objects(text: str) -> list[dict]:
    objects: list[dict] = []
    for candidate in _json_object_strings(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def _json_object_strings(text: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start: int | None = None
    for idx, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : idx + 1])
                start = None
    return spans
