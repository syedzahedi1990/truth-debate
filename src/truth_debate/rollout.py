from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import Task
from .llm import HFGenerator
from .parsing import parse_answer
from .prompts import anti_conformity_update_messages, private_answer_messages, vanilla_update_messages
from .reward import majority_answer


@dataclass
class DebateRollout:
    task_id: str
    protocol: str
    question: str
    gold_answer: str
    category: str
    initial_responses: list[str]
    final_responses: list[str]
    initial_answers: list[str | None]
    final_answers: list[str | None]
    consensus_answer: str | None
    consensus_count: int
    correct: bool
    wrong_consensus: bool
    correct_to_wrong_flip: bool
    answer_diversity: float
    rounds: list[dict[str, Any]]
    meta: dict[str, Any]


def run_single(model: HFGenerator, task: Task, debate_cfg: dict[str, Any]) -> DebateRollout:
    response = model.generate(
        private_answer_messages(task),
        max_new_tokens=int(debate_cfg["max_new_tokens"]),
        temperature=float(debate_cfg["temperature"]),
        top_p=float(debate_cfg["top_p"]),
    )
    return _make_rollout(task, "single", [response], [response], [])


def run_vanilla_debate(model: HFGenerator, task: Task, debate_cfg: dict[str, Any]) -> DebateRollout:
    n_agents = int(debate_cfg["agents"])
    n_rounds = int(debate_cfg["rounds"])
    initial = [
        model.generate(
            private_answer_messages(task, agent_id=i),
            max_new_tokens=int(debate_cfg["max_new_tokens"]),
            temperature=float(debate_cfg["temperature"]),
            top_p=float(debate_cfg["top_p"]),
        )
        for i in range(n_agents)
    ]
    previous = list(initial)
    history: list[dict[str, Any]] = [{"round": 0, "responses": list(previous)}]

    for round_idx in range(1, n_rounds + 1):
        updated: list[str] = []
        for i in range(n_agents):
            peers = [msg for j, msg in enumerate(previous) if j != i]
            updated.append(
                model.generate(
                    vanilla_update_messages(task, previous[i], peers, round_idx, i),
                    max_new_tokens=int(debate_cfg["max_new_tokens"]),
                    temperature=float(debate_cfg["temperature"]),
                    top_p=float(debate_cfg["top_p"]),
                )
            )
        previous = updated
        history.append({"round": round_idx, "responses": list(previous)})

    return _make_rollout(task, "vanilla_debate", initial, previous, history)


def run_anti_conformity_debate(model: HFGenerator, task: Task, debate_cfg: dict[str, Any]) -> DebateRollout:
    n_agents = int(debate_cfg["agents"])
    n_rounds = int(debate_cfg["rounds"])
    commitments = [
        model.generate(
            private_answer_messages(task, agent_id=i),
            max_new_tokens=int(debate_cfg["max_new_tokens"]),
            temperature=float(debate_cfg["temperature"]),
            top_p=float(debate_cfg["top_p"]),
        )
        for i in range(n_agents)
    ]
    previous = list(commitments)
    history: list[dict[str, Any]] = [{"round": 0, "responses": list(previous)}]

    for round_idx in range(1, n_rounds + 1):
        updated: list[str] = []
        for i in range(n_agents):
            peers = [msg for j, msg in enumerate(previous) if j != i]
            updated.append(
                model.generate(
                    anti_conformity_update_messages(task, commitments[i], peers, round_idx, i),
                    max_new_tokens=int(debate_cfg["max_new_tokens"]),
                    temperature=float(debate_cfg["temperature"]),
                    top_p=float(debate_cfg["top_p"]),
                )
            )
        previous = updated
        history.append({"round": round_idx, "responses": list(previous)})

    return _make_rollout(task, "anti_conformity", commitments, previous, history)


def run_protocol(model: HFGenerator, protocol: str, tasks: list[Task], debate_cfg: dict[str, Any]) -> list[DebateRollout]:
    rollouts: list[DebateRollout] = []
    for task in tqdm(tasks, desc=f"eval:{protocol}"):
        if protocol == "single":
            rollouts.append(run_single(model, task, debate_cfg))
        elif protocol == "vanilla_debate":
            rollouts.append(run_vanilla_debate(model, task, debate_cfg))
        elif protocol == "anti_conformity":
            rollouts.append(run_anti_conformity_debate(model, task, debate_cfg))
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    return rollouts


def write_rollouts(path: str | Path, rollouts: list[DebateRollout]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rollout in rollouts:
            f.write(json.dumps(asdict(rollout), sort_keys=True) + "\n")


def read_rollouts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _make_rollout(
    task: Task,
    protocol: str,
    initial_responses: list[str],
    final_responses: list[str],
    rounds: list[dict[str, Any]],
) -> DebateRollout:
    initial_answers = [parse_answer(resp) for resp in initial_responses]
    final_answers = [parse_answer(resp) for resp in final_responses]
    consensus, count = majority_answer(final_answers)
    correct = consensus == str(task.answer)
    wrong_consensus = consensus is not None and count >= max(1, (len(final_answers) // 2) + 1) and not correct
    correct_to_wrong = any(init == str(task.answer) and final != str(task.answer) for init, final in zip(initial_answers, final_answers))
    valid = [answer for answer in final_answers if answer is not None]
    diversity = len(set(valid)) / max(1, len(valid))
    return DebateRollout(
        task_id=task.id,
        protocol=protocol,
        question=task.question,
        gold_answer=str(task.answer),
        category=task.category,
        initial_responses=initial_responses,
        final_responses=final_responses,
        initial_answers=initial_answers,
        final_answers=final_answers,
        consensus_answer=consensus,
        consensus_count=count,
        correct=bool(correct),
        wrong_consensus=bool(wrong_consensus),
        correct_to_wrong_flip=bool(correct_to_wrong),
        answer_diversity=float(diversity),
        rounds=rounds,
        meta=task.meta,
    )
