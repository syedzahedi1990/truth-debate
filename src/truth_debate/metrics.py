from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_rollouts(rollouts: list[Any]) -> dict[str, Any]:
    rows = [r if isinstance(r, dict) else r.__dict__ for r in rollouts]
    if not rows:
        return {"n": 0}

    def mean_bool(key: str) -> float:
        return sum(1.0 for row in rows if row.get(key)) / len(rows)

    def mean_float(key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)

    protocol = str(rows[0].get("protocol", "unknown"))
    is_single = protocol == "single" or max((len(row.get("final_answers", [])) for row in rows), default=1) == 1
    summary = {
        "n": len(rows),
        "protocol": protocol,
        "accuracy": mean_bool("correct"),
        "correct_to_wrong_flip_rate": mean_bool("correct_to_wrong_flip"),
        "mean_answer_diversity": mean_float("answer_diversity"),
        "parse_failure_rate": _parse_failure_rate(rows),
        "by_category": {
            cat: _category_summary(cat_rows, is_single)
            for cat, cat_rows in sorted(by_category.items())
        },
    }
    if is_single:
        summary["wrong_answer_rate"] = _wrong_answer_rate(rows)
    else:
        summary["wrong_consensus_rate"] = mean_bool("wrong_consensus")
    return summary


def _parse_failure_rate(rows: list[dict[str, Any]]) -> float:
    failures = 0
    total = 0
    for row in rows:
        for answer in row.get("final_answers", []):
            total += 1
            if answer is None:
                failures += 1
    return failures / max(1, total)


def _wrong_answer_rate(rows: list[dict[str, Any]]) -> float:
    wrong = 0
    for row in rows:
        answers = row.get("final_answers", [])
        answer = answers[0] if answers else row.get("consensus_answer")
        if answer is None:
            continue
        if str(answer) != str(row.get("gold_answer")):
            wrong += 1
    return wrong / max(1, len(rows))


def _category_summary(rows: list[dict[str, Any]], is_single: bool) -> dict[str, Any]:
    summary = {
        "n": len(rows),
        "accuracy": sum(1.0 for row in rows if row.get("correct")) / len(rows),
    }
    if is_single:
        summary["wrong_answer_rate"] = _wrong_answer_rate(rows)
    else:
        summary["wrong_consensus_rate"] = sum(1.0 for row in rows if row.get("wrong_consensus")) / len(rows)
    return summary
