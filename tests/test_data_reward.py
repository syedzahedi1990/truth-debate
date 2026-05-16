import json
import zipfile

from truth_debate.curriculum import (
    arithmetic_rationale,
    plausible_wrong_answer,
    sample_curriculum_case,
    supervised_examples,
    wrong_majority_peers,
)
from truth_debate.data import Task, eval_left_to_right, safe_eval_expr
from truth_debate.parsing import has_required_format, parse_answer, parse_confidence
from truth_debate.rescore import rescore_run
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


def test_strict_answer_parser_cases():
    assert parse_answer("ANSWER: -4408") == "-4408"
    assert parse_answer("Final answer is **40**") == "40"
    assert parse_answer("The final answer is: 28") == "28"
    assert parse_answer("The final answer is:\n28") == "28"
    assert parse_answer("Final Answer: <integer>: 2108") == "2108"
    assert parse_answer("CONFIDENCE: 100%") is None
    assert parse_answer("1. PRIVATE_ANSWER: 34") is None
    assert parse_answer(r"\boxed{915}") == "915"
    assert parse_answer('{"answer": -4408, "confidence": 0.91}') == "-4408"
    assert parse_answer('{"answer": 14.0, "confidence": 0.91}') == "14"
    assert parse_confidence('{"answer": -4408, "confidence": 0.91}') == 0.91
    assert has_required_format('{"answer": -4408, "confidence": 0.91}')
    assert parse_answer('{"answer": 20, "confidence": 0.8}\n{"answer": 14, "confidence": 0.9}') == "14"


def test_wrong_majority_curriculum_uses_plausible_wrong_answer():
    import random

    task = Task(
        id="t1",
        question="Expression: 2 + 3 * 4",
        answer="14",
        category="precedence_trap",
        meta={"left_to_right_answer": "20", "expression": "2 + 3 * 4"},
    )
    rng = random.Random(0)
    wrong = plausible_wrong_answer(task, rng)
    peers = wrong_majority_peers(task, 2, rng)
    examples = supervised_examples(task, 2, rng)

    assert wrong == "20"
    assert all(parse_answer(peer) == "20" for peer in peers)
    assert any(parse_answer(completion) == "14" for _, completion in examples)
    assert arithmetic_rationale("2 + 3 * 4", "14") == "3*4=12; 2 + 12=14"


def test_mixed_curriculum_can_use_model_private_answer():
    import random

    task = Task(
        id="t2",
        question="Expression: 2 + 3 * 4",
        answer="14",
        category="precedence_trap",
        meta={"left_to_right_answer": "20", "expression": "2 + 3 * 4"},
    )
    cfg = {
        "training": {
            "curriculum": {
                "mix": {
                    "oracle_private_wrong_majority": 0.0,
                    "model_private_wrong_majority": 1.0,
                    "model_private_mixed_peers": 0.0,
                }
            }
        }
    }
    own, peers, mode = sample_curriculum_case(
        task=task,
        n_agents=3,
        rng=random.Random(1),
        cfg=cfg,
        model_private_response='{"answer": 14, "confidence": 0.7}',
    )
    assert mode == "model_private_wrong_majority"
    assert parse_answer(own) == "14"
    assert [parse_answer(peer) for peer in peers] == ["20", "20"]


def test_rescore_reads_zip_without_extracting(tmp_path):
    row = {
        "task_id": "eval-00000",
        "protocol": "single",
        "question": "Expression: 2 + 3 * 4",
        "gold_answer": "14",
        "category": "precedence_trap",
        "initial_responses": ["ANSWER: 14 CONFIDENCE: 90"],
        "final_responses": ["ANSWER: 14 CONFIDENCE: 90"],
        "initial_answers": ["90"],
        "final_answers": ["90"],
        "consensus_answer": "90",
        "consensus_count": 1,
        "correct": False,
        "wrong_consensus": True,
        "correct_to_wrong_flip": False,
        "answer_diversity": 1.0,
        "rounds": [],
        "meta": {},
    }
    zip_path = tmp_path / "run.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("run/baseline_rollouts/single.jsonl", json.dumps(row) + "\n")

    out = tmp_path / "out"
    report = rescore_run(zip_path, out)

    assert report == out / "rescored_report.md"
    assert not (tmp_path / "run").exists()
    metrics = json.loads((out / "rescored_metrics" / "baseline_single.json").read_text(encoding="utf-8"))
    assert metrics["accuracy"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["parser_changed_rate"] == 1.0
