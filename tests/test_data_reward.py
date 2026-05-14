from truth_debate.data import Task, eval_left_to_right, safe_eval_expr
from truth_debate.reward import compute_reward


def test_safe_eval_respects_precedence():
    expr = "2 + 3 * 4 - 5"
    assert safe_eval_expr(expr) == 9
    assert eval_left_to_right(expr) == 15


def test_reward_escape_wrong_majority():
    task = Task(
        id="t0",
        question="Expression: 2 + 3 * 4",
        answer="14",
        category="precedence_trap",
        meta={},
    )
    cfg = {
        "correct": 1.0,
        "wrong": -0.2,
        "parse_failure": -0.5,
        "escape_wrong_majority": 0.6,
        "join_wrong_majority": -0.5,
        "correct_to_wrong_flip": -0.8,
        "format_bonus": 0.05,
        "confidence_calibration": 0.1,
    }
    reward = compute_reward(
        task=task,
        final_response="ERROR_CHECK: multiplication first. ANSWER: 14 CONFIDENCE: 90",
        peer_responses=["ANSWER: 20 CONFIDENCE: 70", "ANSWER: 20 CONFIDENCE: 80"],
        own_initial_response="ANSWER: 14 CONFIDENCE: 60",
        reward_cfg=cfg,
    )
    assert reward.parsed_answer == "14"
    assert reward.majority_is_wrong
    assert reward.anti_conformity_component > 0
    assert reward.total > 1.0
