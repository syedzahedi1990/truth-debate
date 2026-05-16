import json
import zipfile

from truth_debate.data import Task, eval_left_to_right, safe_eval_expr
from truth_debate.parsing import parse_answer
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
