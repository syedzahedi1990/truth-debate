from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "model": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "auto",
        "load_in_4bit": False,
        "trust_remote_code": False,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "data": {
        "kind": "synthetic_arithmetic",
        "train_size": 100,
        "eval_size": 50,
        "max_value": 50,
        "min_terms": 4,
        "max_terms": 7,
        "trap_fraction": 0.7,
    },
    "debate": {
        "protocols": ["single", "vanilla_debate", "anti_conformity"],
        "agents": 3,
        "rounds": 2,
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.9,
    },
    "reward": {
        "correct": 1.0,
        "wrong": -0.2,
        "parse_failure": -0.5,
        "escape_wrong_majority": 0.5,
        "join_wrong_majority": -0.4,
        "correct_to_wrong_flip": -0.7,
        "format_bonus": 0.05,
        "confidence_calibration": 0.1,
    },
    "training": {
        "enabled": True,
        "sft_warmup_steps": 0,
        "steps": 100,
        "batch_size": 1,
        "lr": 3e-5,
        "grad_clip": 1.0,
        "baseline_ema": 0.95,
        "save_every": 50,
        "save_post_sft": True,
        "sft_anchor_weight": 0.0,
        "advantage_clip": None,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "curriculum": {
            "wrong_majority": False,
            "mix": {
                "oracle_private_wrong_majority": 0.4,
                "model_private_wrong_majority": 0.4,
                "model_private_mixed_peers": 0.2,
            },
        },
    },
    "evaluation": {
        "max_tasks": None,
        "evaluate_post_sft": False,
        "evaluate_trained": True,
    },
}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    return cfg


def write_resolved_config(cfg: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
