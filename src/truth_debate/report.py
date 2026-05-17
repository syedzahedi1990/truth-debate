from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_report(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    report_path = output / "report.md"
    metric_files = sorted(metrics_dir.glob("*.json"))

    sections = ["# Truth-Seeking Debate Report", ""]
    if not metric_files:
        sections.append("No metrics found.")
        report_path.write_text("\n".join(sections), encoding="utf-8")
        return report_path

    grouped: dict[str, Any] = {}
    for path in metric_files:
        with open(path, "r", encoding="utf-8") as f:
            grouped[path.stem] = json.load(f)

    for name, metrics in grouped.items():
        if not isinstance(metrics, dict) or "accuracy" not in metrics:
            continue
        sections.extend(
            [
                f"## {name}",
                "",
                f"- n: {metrics.get('n', 0)}",
                f"- accuracy: {_pct(metrics.get('accuracy'))}",
                _wrong_line(metrics),
                f"- correct-to-wrong flip rate: {_pct(metrics.get('correct_to_wrong_flip_rate'))}",
                f"- parse failure rate: {_pct(metrics.get('parse_failure_rate'))}",
                *_standard_numeric_lines(metrics),
                f"- mean answer diversity: {metrics.get('mean_answer_diversity', 0):.3f}",
                "",
            ]
        )
        by_category = metrics.get("by_category", {})
        if by_category:
            wrong_header = "wrong answer" if "wrong_answer_rate" in metrics else "wrong consensus"
            wrong_key = "wrong_answer_rate" if "wrong_answer_rate" in metrics else "wrong_consensus_rate"
            sections.append(f"| category | n | accuracy | {wrong_header} |")
            sections.append("| --- | ---: | ---: | ---: |")
            for category, row in by_category.items():
                sections.append(
                    f"| {category} | {row.get('n', 0)} | {_pct(row.get('accuracy'))} | "
                    f"{_pct(row.get(wrong_key))} |"
                )
            sections.append("")

    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _wrong_line(metrics: dict[str, Any]) -> str:
    if "wrong_answer_rate" in metrics:
        return f"- wrong answer rate: {_pct(metrics.get('wrong_answer_rate'))}"
    return f"- wrong consensus rate: {_pct(metrics.get('wrong_consensus_rate'))}"


def _standard_numeric_lines(metrics: dict[str, Any]) -> list[str]:
    if "standard_numeric_accuracy" not in metrics:
        return []
    return [
        f"- standard numeric accuracy: {_pct(metrics.get('standard_numeric_accuracy'))}",
        f"- standard numeric parse failure rate: {_pct(metrics.get('standard_numeric_parse_failure_rate'))}",
    ]
