from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import write_json
from .data import build_datasets, read_tasks
from .llm import HFGenerator
from .metrics import summarize_rollouts
from .rollout import run_protocol, write_rollouts


def run_evaluation(
    cfg: dict[str, Any],
    output_dir: str | Path,
    label: str,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    eval_path = output / "data" / "eval.jsonl"
    if not eval_path.exists():
        build_datasets(cfg, output)

    max_tasks = cfg.get("evaluation", {}).get("max_tasks")
    tasks = read_tasks(eval_path, max_tasks=max_tasks)
    model = HFGenerator(cfg["model"], adapter_path=adapter_path)

    all_metrics: dict[str, Any] = {}
    rollout_dir = output / f"{label}_rollouts"
    metrics_dir = output / "metrics"
    for protocol in cfg["debate"]["protocols"]:
        rollouts = run_protocol(model, protocol, tasks, cfg["debate"])
        write_rollouts(rollout_dir / f"{protocol}.jsonl", rollouts)
        metrics = summarize_rollouts(rollouts)
        all_metrics[protocol] = metrics
        write_json(metrics_dir / f"{label}_{protocol}.json", metrics)

    write_json(metrics_dir / f"{label}_all.json", all_metrics)
    return all_metrics
