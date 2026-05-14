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

    return {
        "n": len(rows),
        "accuracy": mean_bool("correct"),
        "wrong_consensus_rate": mean_bool("wrong_consensus"),
        "correct_to_wrong_flip_rate": mean_bool("correct_to_wrong_flip"),
        "mean_answer_diversity": mean_float("answer_diversity"),
        "parse_failure_rate": _parse_failure_rate(rows),
        "by_category": {
            cat: {
                "n": len(cat_rows),
                "accuracy": sum(1.0 for row in cat_rows if row.get("correct")) / len(cat_rows),
                "wrong_consensus_rate": sum(1.0 for row in cat_rows if row.get("wrong_consensus")) / len(cat_rows),
            }
            for cat, cat_rows in sorted(by_category.items())
        },
    }


def _parse_failure_rate(rows: list[dict[str, Any]]) -> float:
    failures = 0
    total = 0
    for row in rows:
        for answer in row.get("final_answers", []):
            total += 1
            if answer is None:
                failures += 1
    return failures / max(1, total)
