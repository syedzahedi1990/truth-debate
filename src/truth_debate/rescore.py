from __future__ import annotations

import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .parsing import parse_answer, parse_answer_legacy
from .reward import majority_answer


PROTOCOL_ORDER = {"single": 0, "vanilla_debate": 1, "anti_conformity": 2}
LABEL_ORDER = {"baseline": 0, "trained": 1}


def rescore_run(source: str | Path, output_dir: str | Path | None = None) -> Path:
    source_path = Path(source)
    output = Path(output_dir) if output_dir else _default_output_dir(source_path)
    metrics_dir = output / "rescored_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with RunSource(source_path) as run_source:
        all_metrics: dict[str, dict[str, Any]] = {}
        for label, protocol, entry_name in run_source.rollout_entries():
            rows = [rescore_rollout(row, protocol) for row in run_source.read_jsonl(entry_name)]
            metrics = summarize_rescored_rollouts(rows)
            all_metrics.setdefault(label, {})[protocol] = metrics
            _write_json(metrics_dir / f"{label}_{protocol}.json", metrics)

        for label, metrics in all_metrics.items():
            _write_json(metrics_dir / f"{label}_all.json", metrics)

        reward_summary = summarize_reward_log(run_source)

    report_path = build_rescored_report(output, source_path, all_metrics, reward_summary)
    print(f"Wrote {report_path}")
    return report_path


class RunSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive: zipfile.ZipFile | None = None

    def __enter__(self) -> "RunSource":
        if self.path.suffix.lower() == ".zip":
            self.archive = zipfile.ZipFile(self.path)
        elif not self.path.is_dir():
            raise FileNotFoundError(f"Run source must be a directory or .zip file: {self.path}")
        return self

    def __exit__(self, *_args: object) -> None:
        if self.archive:
            self.archive.close()

    def names(self) -> list[str]:
        if self.archive:
            return [name for name in self.archive.namelist() if not name.endswith("/")]
        return [path.relative_to(self.path).as_posix() for path in self.path.rglob("*") if path.is_file()]

    def rollout_entries(self) -> list[tuple[str, str, str]]:
        entries: list[tuple[str, str, str]] = []
        for name in self.names():
            parts = name.replace("\\", "/").split("/")
            for idx, part in enumerate(parts[:-1]):
                if not part.endswith("_rollouts"):
                    continue
                protocol = Path(parts[idx + 1]).stem
                if protocol not in PROTOCOL_ORDER:
                    continue
                label = part[: -len("_rollouts")]
                entries.append((label, protocol, name))
        return sorted(entries, key=lambda item: (LABEL_ORDER.get(item[0], 99), PROTOCOL_ORDER.get(item[1], 99), item[2]))

    def read_text(self, name: str) -> str:
        if self.archive:
            with self.archive.open(name) as f:
                return f.read().decode("utf-8")
        return (self.path / name).read_text(encoding="utf-8")

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.read_text(name).splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows


def rescore_rollout(row: dict[str, Any], protocol: str | None = None) -> dict[str, Any]:
    protocol = protocol or str(row.get("protocol", "unknown"))
    gold = str(row["gold_answer"])
    old_initial = list(row.get("initial_answers", []))
    old_final = list(row.get("final_answers", []))
    initial_responses = list(row.get("initial_responses", []))
    final_responses = list(row.get("final_responses", []))

    initial_answers = [parse_answer(text) for text in initial_responses]
    final_answers = [parse_answer(text) for text in final_responses]
    consensus, count = majority_answer(final_answers)
    is_single = protocol == "single" or len(final_answers) == 1
    correct = final_answers[0] == gold if is_single and final_answers else consensus == gold
    wrong_consensus = (
        not is_single
        and consensus is not None
        and count >= max(1, (len(final_answers) // 2) + 1)
        and not correct
    )
    wrong_answer = is_single and bool(final_answers) and final_answers[0] is not None and not correct
    correct_to_wrong = any(init == gold and final != gold for init, final in zip(initial_answers, final_answers))
    valid = [answer for answer in final_answers if answer is not None]

    rescored = dict(row)
    rescored.update(
        {
            "protocol": protocol,
            "old_initial_answers": old_initial,
            "old_final_answers": old_final,
            "old_consensus_answer": row.get("consensus_answer"),
            "initial_answers": initial_answers,
            "final_answers": final_answers,
            "consensus_answer": consensus,
            "consensus_count": count,
            "correct": bool(correct),
            "wrong_answer": bool(wrong_answer),
            "wrong_consensus": bool(wrong_consensus),
            "correct_to_wrong_flip": bool(correct_to_wrong),
            "answer_diversity": len(set(valid)) / max(1, len(valid)),
            "parser_changed": old_final != final_answers,
            "legacy_reparse_final_answers": [parse_answer_legacy(text) for text in final_responses],
        }
    )
    return rescored


def summarize_rescored_rollouts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    protocol = str(rows[0].get("protocol", "unknown"))
    is_single = protocol == "single" or max((len(row.get("final_answers", [])) for row in rows), default=1) == 1
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)

    summary: dict[str, Any] = {
        "n": len(rows),
        "protocol": protocol,
        "accuracy": _mean_bool(rows, "correct"),
        "correct_to_wrong_flip_rate": _mean_bool(rows, "correct_to_wrong_flip"),
        "parse_failure_rate": _parse_failure_rate(rows),
        "mean_answer_diversity": _mean_float(rows, "answer_diversity"),
        "parser_changed_rate": _mean_bool(rows, "parser_changed"),
        "top_parsed_answers": _top_answers(rows),
        "diagnostics": {
            "changed_examples": _changed_examples(rows),
            "unparseable_examples": _unparseable_examples(rows),
        },
        "by_category": {
            category: _category_summary(category_rows, is_single)
            for category, category_rows in sorted(by_category.items())
        },
    }
    if is_single:
        summary["wrong_answer_rate"] = _mean_bool(rows, "wrong_answer")
    else:
        summary["wrong_consensus_rate"] = _mean_bool(rows, "wrong_consensus")
    return summary


def summarize_reward_log(source: RunSource) -> dict[str, Any] | None:
    log_name = next((name for name in source.names() if name.endswith("logs/rl_train.jsonl")), None)
    if not log_name:
        return None

    rewards: list[dict[str, Any]] = []
    for row in source.read_jsonl(log_name):
        reward = row.get("reward")
        if isinstance(reward, dict):
            reward = dict(reward)
            reward["step"] = row.get("step")
            rewards.append(reward)
    if not rewards:
        return None

    anti = [float(row.get("anti_conformity_component", 0.0)) for row in rewards]
    totals = [float(row.get("total", 0.0)) for row in rewards]
    correct = [float(row.get("correct_component", 0.0)) for row in rewards]
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rewards:
        step = int(row.get("step") or 0)
        bucket = ((max(step, 1) - 1) // 100 + 1) * 100
        buckets[bucket].append(row)

    return {
        "n": len(rewards),
        "mean_total_reward": statistics.mean(totals),
        "mean_correct_component": statistics.mean(correct),
        "mean_anti_conformity_component": statistics.mean(anti),
        "anti_conformity_nonzero_rate": sum(1 for value in anti if value != 0.0) / len(anti),
        "by_100_step_bucket": {
            str(bucket): {
                "n": len(items),
                "mean_total_reward": statistics.mean(float(item.get("total", 0.0)) for item in items),
                "mean_anti_conformity_component": statistics.mean(
                    float(item.get("anti_conformity_component", 0.0)) for item in items
                ),
            }
            for bucket, items in sorted(buckets.items())
        },
    }


def build_rescored_report(
    output_dir: str | Path,
    source: Path,
    all_metrics: dict[str, dict[str, Any]],
    reward_summary: dict[str, Any] | None,
) -> Path:
    output = Path(output_dir)
    report_path = output / "rescored_report.md"
    sections = [
        "# Rescored Truth-Seeking Debate Report",
        "",
        f"Source: `{source}`",
        "",
        "## Diagnosis",
        "",
        "- The original run completed, but the original parser used a last-integer fallback that can read confidence values, numbered list markers, or peer/private answers as final answers.",
        "- The rescored metrics below use only explicit final-answer markers. Ambiguous outputs are counted as parse failures.",
        "- Treat this run as a smoke test and diagnostic dataset unless accuracy remains stable under strict parsing.",
        "",
    ]

    if reward_summary:
        sections.extend(
            [
                "## RL Reward Diagnostics",
                "",
                f"- training samples: {reward_summary['n']}",
                f"- mean total reward: {reward_summary['mean_total_reward']:.4f}",
                f"- mean correctness component: {reward_summary['mean_correct_component']:.4f}",
                f"- mean anti-conformity component: {reward_summary['mean_anti_conformity_component']:.4f}",
                f"- anti-conformity nonzero rate: {_pct(reward_summary['anti_conformity_nonzero_rate'])}",
                "",
            ]
        )

    for label, label_metrics in sorted(all_metrics.items(), key=lambda item: LABEL_ORDER.get(item[0], 99)):
        sections.extend([f"## {label.title()} Metrics", ""])
        for protocol, metrics in sorted(label_metrics.items(), key=lambda item: PROTOCOL_ORDER.get(item[0], 99)):
            sections.extend(_metric_section(protocol, metrics))

    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


def _metric_section(protocol: str, metrics: dict[str, Any]) -> list[str]:
    wrong_line = (
        f"- wrong answer rate: {_pct(metrics.get('wrong_answer_rate'))}"
        if "wrong_answer_rate" in metrics
        else f"- wrong consensus rate: {_pct(metrics.get('wrong_consensus_rate'))}"
    )
    lines = [
        f"### {protocol}",
        "",
        f"- n: {metrics.get('n', 0)}",
        f"- accuracy: {_pct(metrics.get('accuracy'))}",
        wrong_line,
        f"- parse failure rate: {_pct(metrics.get('parse_failure_rate'))}",
        f"- correct-to-wrong flip rate: {_pct(metrics.get('correct_to_wrong_flip_rate'))}",
        f"- parser changed rate: {_pct(metrics.get('parser_changed_rate'))}",
        f"- mean answer diversity: {metrics.get('mean_answer_diversity', 0.0):.3f}",
        f"- top parsed answers: {_format_top_answers(metrics.get('top_parsed_answers', []))}",
        "",
    ]
    by_category = metrics.get("by_category", {})
    if by_category:
        wrong_header = "wrong answer" if "wrong_answer_rate" in metrics else "wrong consensus"
        wrong_key = "wrong_answer_rate" if "wrong_answer_rate" in metrics else "wrong_consensus_rate"
        lines.append(f"| category | n | accuracy | {wrong_header} | parse failure |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for category, row in by_category.items():
            lines.append(
                f"| {category} | {row.get('n', 0)} | {_pct(row.get('accuracy'))} | "
                f"{_pct(row.get(wrong_key))} | {_pct(row.get('parse_failure_rate'))} |"
            )
        lines.append("")

    diagnostics = metrics.get("diagnostics", {})
    changed = diagnostics.get("changed_examples", [])
    unparseable = diagnostics.get("unparseable_examples", [])
    if changed:
        lines.extend(["Changed parser examples:", ""])
        for example in changed[:3]:
            lines.append(
                f"- `{example['task_id']}` gold `{example['gold_answer']}`: old `{example['old_final_answers']}` -> "
                f"new `{example['new_final_answers']}`; excerpt: {example['excerpt']}"
            )
        lines.append("")
    if unparseable:
        lines.extend(["Unparseable examples:", ""])
        for example in unparseable[:3]:
            lines.append(
                f"- `{example['task_id']}` response {example['response_index']} gold `{example['gold_answer']}`; "
                f"excerpt: {example['excerpt']}"
            )
        lines.append("")
    return lines


def _category_summary(rows: list[dict[str, Any]], is_single: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": len(rows),
        "accuracy": _mean_bool(rows, "correct"),
        "parse_failure_rate": _parse_failure_rate(rows),
    }
    if is_single:
        summary["wrong_answer_rate"] = _mean_bool(rows, "wrong_answer")
    else:
        summary["wrong_consensus_rate"] = _mean_bool(rows, "wrong_consensus")
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


def _top_answers(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        answers = row.get("final_answers", [])
        consensus = row.get("consensus_answer")
        value = answers[0] if len(answers) == 1 else consensus
        counts["parse_failure" if value is None else str(value)] += 1
    return [{"answer": answer, "count": count} for answer, count in counts.most_common(limit)]


def _changed_examples(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("parser_changed"):
            continue
        examples.append(
            {
                "task_id": row.get("task_id"),
                "gold_answer": row.get("gold_answer"),
                "old_final_answers": row.get("old_final_answers"),
                "new_final_answers": row.get("final_answers"),
                "old_consensus_answer": row.get("old_consensus_answer"),
                "new_consensus_answer": row.get("consensus_answer"),
                "excerpt": _first_changed_excerpt(row),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _unparseable_examples(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        for idx, answer in enumerate(row.get("final_answers", [])):
            if answer is not None:
                continue
            responses = row.get("final_responses", [])
            examples.append(
                {
                    "task_id": row.get("task_id"),
                    "gold_answer": row.get("gold_answer"),
                    "response_index": idx,
                    "excerpt": _excerpt(responses[idx] if idx < len(responses) else ""),
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def _first_changed_excerpt(row: dict[str, Any]) -> str:
    old = row.get("old_final_answers", [])
    new = row.get("final_answers", [])
    responses = row.get("final_responses", [])
    for idx, (old_answer, new_answer) in enumerate(zip(old, new)):
        if old_answer != new_answer and idx < len(responses):
            return _excerpt(responses[idx])
    return _excerpt(responses[0] if responses else "")


def _excerpt(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text).replace("`", "'").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def _mean_bool(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(1.0 for row in rows if row.get(key)) / max(1, len(rows))


def _mean_float(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(float(row.get(key, 0.0)) for row in rows) / max(1, len(rows))


def _format_top_answers(values: list[dict[str, Any]]) -> str:
    if not values:
        return "n/a"
    return ", ".join(f"{item['answer']} ({item['count']})" for item in values[:5])


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _default_output_dir(source: Path) -> Path:
    if source.suffix.lower() == ".zip":
        return source.with_name(f"{source.stem}_rescored")
    return source
