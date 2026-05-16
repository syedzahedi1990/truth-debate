from __future__ import annotations

import ast
import json
import operator
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    category: str
    meta: dict


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}


def safe_eval_expr(expr: str) -> int:
    node = ast.parse(expr, mode="eval")

    def visit(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return int(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -visit(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(n.op)](visit(n.left), visit(n.right))
        raise ValueError(f"Unsupported expression node: {ast.dump(n)}")

    return visit(node)


def eval_left_to_right(expr: str) -> int:
    parts = expr.split()
    total = int(parts[0])
    idx = 1
    while idx < len(parts):
        op = parts[idx]
        rhs = int(parts[idx + 1])
        if op == "+":
            total += rhs
        elif op == "-":
            total -= rhs
        elif op == "*":
            total *= rhs
        else:
            raise ValueError(f"Unknown operator: {op}")
        idx += 2
    return total


def _make_expression(rng: random.Random, min_terms: int, max_terms: int, max_value: int, force_trap: bool) -> str:
    n_terms = rng.randint(min_terms, max_terms)
    nums = [str(rng.randint(0, max_value)) for _ in range(n_terms)]
    if force_trap:
        ops = [rng.choice(["+", "-"]) for _ in range(n_terms - 1)]
        mult_idx = rng.randrange(n_terms - 1)
        ops[mult_idx] = "*"
        if n_terms > 3:
            ops[rng.randrange(n_terms - 1)] = rng.choice(["+", "-"])
    else:
        ops = [rng.choice(["+", "-", "*"]) for _ in range(n_terms - 1)]

    pieces: list[str] = []
    for i, num in enumerate(nums):
        pieces.append(num)
        if i < len(ops):
            pieces.append(ops[i])
    return " ".join(pieces)


def generate_arithmetic_tasks(
    n: int,
    seed: int,
    split: str,
    max_value: int = 50,
    min_terms: int = 4,
    max_terms: int = 7,
    trap_fraction: float = 0.7,
) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen: set[str] = set()

    while len(tasks) < n:
        force_trap = rng.random() < trap_fraction
        expr = _make_expression(rng, min_terms, max_terms, max_value, force_trap)
        if expr in seen:
            continue
        seen.add(expr)

        true_answer = safe_eval_expr(expr)
        ltr_answer = eval_left_to_right(expr)
        category = "precedence_trap" if true_answer != ltr_answer else "arithmetic"
        question = (
            "Compute this expression exactly using standard mathematical order of operations. "
            'Return the final integer as JSON with an "answer" field.\n\n'
            f"Expression: {expr}"
        )
        tasks.append(
            Task(
                id=f"{split}-{len(tasks):05d}",
                question=question,
                answer=str(true_answer),
                category=category,
                meta={
                    "expression": expr,
                    "left_to_right_answer": str(ltr_answer),
                    "is_trap": true_answer != ltr_answer,
                },
            )
        )
    return tasks


def build_datasets(cfg: dict, output_dir: str | Path) -> tuple[Path, Path]:
    data_cfg = cfg["data"]
    seed = int(cfg["seed"])
    out = Path(output_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)

    kind = str(data_cfg.get("kind", "synthetic_arithmetic"))
    if kind == "synthetic_arithmetic":
        train = generate_arithmetic_tasks(split="train", seed=seed, n=int(data_cfg["train_size"]), **_task_kwargs(data_cfg))
        eval_tasks = generate_arithmetic_tasks(split="eval", seed=seed + 1, n=int(data_cfg["eval_size"]), **_task_kwargs(data_cfg))
    elif kind == "gsm8k":
        train = load_gsm8k(split="train", n=int(data_cfg["train_size"]), seed=seed)
        eval_tasks = load_gsm8k(split="test", n=int(data_cfg["eval_size"]), seed=seed + 1)
    else:
        raise ValueError(f"Unknown data.kind: {kind}")

    train_path = out / "train.jsonl"
    eval_path = out / "eval.jsonl"
    write_tasks(train_path, train)
    write_tasks(eval_path, eval_tasks)
    return train_path, eval_path


def _task_kwargs(data_cfg: dict) -> dict:
    return {
        "max_value": int(data_cfg["max_value"]),
        "min_terms": int(data_cfg["min_terms"]),
        "max_terms": int(data_cfg["max_terms"]),
        "trap_fraction": float(data_cfg["trap_fraction"]),
    }


def write_tasks(path: str | Path, tasks: Iterable[Task]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), sort_keys=True) + "\n")


def read_tasks(path: str | Path, max_tasks: int | None = None) -> list[Task]:
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(Task(**obj))
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
    return tasks


def load_gsm8k(split: str, n: int, seed: int) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    tasks: list[Task] = []
    for idx, row in enumerate(ds):
        answer = _extract_gsm8k_answer(row["answer"])
        question = (
            'Solve this grade-school math problem. Return the final integer as JSON with an "answer" field.\n\n'
            f"Problem: {row['question']}"
        )
        tasks.append(
            Task(
                id=f"gsm8k-{split}-{idx:05d}",
                question=question,
                answer=answer,
                category="gsm8k",
                meta={"source": "gsm8k", "answer_rationale": row["answer"]},
            )
        )
    return tasks


def _extract_gsm8k_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        raw = answer_text.split(marker)[-1].strip().replace(",", "")
        return str(int(float(raw)))
    nums = [chunk.replace(",", "") for chunk in answer_text.split() if chunk.replace(",", "").lstrip("-").isdigit()]
    if not nums:
        raise ValueError(f"Could not parse GSM8K answer: {answer_text}")
    return str(int(nums[-1]))
