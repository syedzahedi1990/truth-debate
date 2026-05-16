from __future__ import annotations

import random
import re
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
    return json_private_answer(task.answer, confidence=0.95, rationale=computation_rationale(task))


def supervised_examples(task: Task, n_peers: int, rng: random.Random) -> list[tuple[list[dict[str, str]], str]]:
    wrong = plausible_wrong_answer(task, rng)
    own = oracle_private_answer(task)
    peers = [json_private_answer(wrong, confidence=0.82, rationale="plausible but wrong majority shortcut") for _ in range(n_peers)]
    return [
        (
            private_answer_messages(task, agent_id=0),
            json_private_answer(task.answer, confidence=0.95, rationale=computation_rationale(task)),
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


def computation_rationale(task: Task) -> str:
    expression = task.meta.get("expression")
    if expression:
        return arithmetic_rationale(str(expression), str(task.answer))

    rationale = str(task.meta.get("answer_rationale", "")).strip()
    if rationale:
        rationale = rationale.replace("\n", " ")
        rationale = rationale.split("####")[0].strip()
        rationale = re.sub(r"\s+", " ", rationale)
        return rationale[:240] or "solve step by step and return the final integer"
    return "solve step by step and return the final integer"


def arithmetic_rationale(expression: str, answer: str) -> str:
    parts = expression.split()
    if not parts:
        return f"answer is {answer}"

    groups: list[tuple[str, list[int], int]] = []
    sign = "+"
    factors = [int(parts[0])]
    idx = 1
    while idx < len(parts):
        op = parts[idx]
        rhs = int(parts[idx + 1])
        if op == "*":
            factors.append(rhs)
        else:
            product = _product(factors)
            groups.append((sign, factors, product))
            sign = op
            factors = [rhs]
        idx += 2
    groups.append((sign, factors, _product(factors)))

    mult_steps = []
    simplified: list[str] = []
    for group_idx, (sign, factors, product) in enumerate(groups):
        if len(factors) > 1:
            mult_steps.append(f"{'*'.join(str(value) for value in factors)}={product}")
        prefix = "" if group_idx == 0 and sign == "+" else f"{sign} "
        simplified.append(f"{prefix}{product}")
    simplified_expr = " ".join(simplified)
    if mult_steps:
        return f"{'; '.join(mult_steps)}; {simplified_expr}={answer}"
    return f"evaluate left to right: {expression}={answer}"


def sample_curriculum_case(
    task: Task,
    n_agents: int,
    rng: random.Random,
    cfg: dict[str, Any],
    model_private_response: str | None = None,
) -> tuple[str, list[str], str]:
    curriculum = cfg.get("training", {}).get("curriculum", {})
    mix = curriculum.get("mix") or {
        "oracle_private_wrong_majority": 0.4,
        "model_private_wrong_majority": 0.4,
        "model_private_mixed_peers": 0.2,
    }
    mode = _weighted_choice(mix, rng)
    n_peers = max(0, n_agents - 1)
    wrong = plausible_wrong_answer(task, rng)

    if mode == "oracle_private_wrong_majority":
        return oracle_private_answer(task), wrong_majority_peers(task, n_peers, rng), mode

    if mode == "model_private_wrong_majority":
        own = model_private_response or oracle_private_answer(task)
        return own, wrong_majority_peers(task, n_peers, rng), mode

    if mode == "model_private_mixed_peers":
        own = model_private_response or oracle_private_answer(task)
        peers = [json_private_answer(wrong, confidence=0.78, rationale="plausible but wrong shortcut")]
        if n_peers > 1:
            peers.append(oracle_private_answer(task))
        while len(peers) < n_peers:
            peers.append(json_private_answer(wrong, confidence=0.65, rationale="plausible but wrong shortcut"))
        return own, peers[:n_peers], mode

    return oracle_private_answer(task), wrong_majority_peers(task, n_peers, rng), "oracle_private_wrong_majority"


def curriculum_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("curriculum", {}).get("wrong_majority", False))


def _product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def _weighted_choice(weights: dict[str, float], rng: random.Random) -> str:
    items = [(key, max(0.0, float(value))) for key, value in weights.items()]
    total = sum(value for _, value in items)
    if total <= 0:
        return "oracle_private_wrong_majority"
    pick = rng.random() * total
    running = 0.0
    for key, value in items:
        running += value
        if pick <= running:
            return key
    return items[-1][0]
