from __future__ import annotations

import random
from typing import Any

from .data import Task
from .prompts import json_anti_conformity_answer, json_private_answer, private_answer_messages, rl_final_messages


def plausible_wrong_answer(task: Task, rng: random.Random) -> str:
    gold = int(task.answer)
    ltr = task.meta.get("left_to_right_answer")
    if ltr is not None and str(ltr) != str(task.answer):
        return str(int(ltr))

    expression = task.meta.get("expression", "")
    nums = [int(part) for part in str(expression).split() if part.lstrip("-").isdigit()]
    candidates = [
        gold + rng.choice([-10, -5, -2, -1, 1, 2, 5, 10]),
        -gold,
        sum(nums) if nums else gold + 1,
    ]
    for candidate in candidates:
        if candidate != gold:
            return str(candidate)
    return str(gold + 1)


def wrong_majority_peers(task: Task, n_peers: int, rng: random.Random) -> list[str]:
    wrong = plausible_wrong_answer(task, rng)
    peers: list[str] = []
    for _ in range(n_peers):
        peers.append(json_private_answer(wrong, confidence=0.82, rationale="plausible but wrong majority shortcut"))
    return peers


def oracle_private_answer(task: Task) -> str:
    return json_private_answer(task.answer, confidence=0.95, rationale="standard order of operations")


def supervised_examples(task: Task, n_peers: int, rng: random.Random) -> list[tuple[list[dict[str, str]], str]]:
    wrong = plausible_wrong_answer(task, rng)
    own = oracle_private_answer(task)
    peers = [json_private_answer(wrong, confidence=0.82, rationale="plausible but wrong majority shortcut") for _ in range(n_peers)]
    return [
        (
            private_answer_messages(task, agent_id=0),
            json_private_answer(task.answer, confidence=0.95, rationale="standard order of operations"),
        ),
        (
            rl_final_messages(task, own, peers),
            json_anti_conformity_answer(
                answer=task.answer,
                private_answer=task.answer,
                wrong_majority_answer=wrong,
                confidence=0.95,
            ),
        ),
    ]


def curriculum_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("curriculum", {}).get("wrong_majority", False))
