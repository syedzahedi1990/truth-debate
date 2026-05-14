from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import Task
from .parsing import has_required_format, parse_answer, parse_confidence


@dataclass
class RewardBreakdown:
    total: float
    correct_component: float
    anti_conformity_component: float
    flip_component: float
    format_component: float
    calibration_component: float
    parsed_answer: str | None
    majority_answer: str | None
    majority_is_wrong: bool


def majority_answer(answers: list[str | None]) -> tuple[str | None, int]:
    counts: dict[str, int] = {}
    for answer in answers:
        if answer is None:
            continue
        counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None, 0
    answer, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return answer, count


def compute_reward(
    task: Task,
    final_response: str,
    peer_responses: list[str],
    own_initial_response: str | None,
    reward_cfg: dict[str, Any],
) -> RewardBreakdown:
    gold = str(task.answer)
    parsed_final = parse_answer(final_response)
    peer_answers = [parse_answer(resp) for resp in peer_responses]
    maj_answer, maj_count = majority_answer(peer_answers)
    majority_is_wrong = maj_answer is not None and maj_count >= 2 and maj_answer != gold

    if parsed_final is None:
        correct_component = float(reward_cfg["parse_failure"])
    elif parsed_final == gold:
        correct_component = float(reward_cfg["correct"])
    else:
        correct_component = float(reward_cfg["wrong"])

    anti_component = 0.0
    if majority_is_wrong and parsed_final == gold and parsed_final != maj_answer:
        anti_component += float(reward_cfg["escape_wrong_majority"])
    if majority_is_wrong and parsed_final == maj_answer:
        anti_component += float(reward_cfg["join_wrong_majority"])

    flip_component = 0.0
    if own_initial_response is not None:
        own_initial = parse_answer(own_initial_response)
        if own_initial == gold and parsed_final not in (gold, None):
            flip_component += float(reward_cfg["correct_to_wrong_flip"])

    format_component = float(reward_cfg["format_bonus"]) if has_required_format(final_response) else 0.0

    calibration_component = 0.0
    conf = parse_confidence(final_response)
    if conf is not None:
        is_correct = parsed_final == gold
        if is_correct:
            calibration_component = float(reward_cfg["confidence_calibration"]) * conf
        else:
            calibration_component = -float(reward_cfg["confidence_calibration"]) * conf

    total = correct_component + anti_component + flip_component + format_component + calibration_component
    return RewardBreakdown(
        total=float(total),
        correct_component=float(correct_component),
        anti_conformity_component=float(anti_component),
        flip_component=float(flip_component),
        format_component=float(format_component),
        calibration_component=float(calibration_component),
        parsed_answer=parsed_final,
        majority_answer=maj_answer,
        majority_is_wrong=bool(majority_is_wrong),
    )
