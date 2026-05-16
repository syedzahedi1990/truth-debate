from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import trange

from .curriculum import curriculum_enabled, oracle_private_answer, supervised_examples, wrong_majority_peers
from .data import build_datasets, read_tasks
from .llm import HFGenerator
from .prompts import private_answer_messages, rl_final_messages
from .reward import compute_reward


def run_rl_training(cfg: dict[str, Any], output_dir: str | Path) -> Path:
    import torch

    output = Path(output_dir)
    train_path = output / "data" / "train.jsonl"
    if not train_path.exists():
        build_datasets(cfg, output)
    tasks = read_tasks(train_path)

    train_cfg = cfg["training"]
    model = HFGenerator(
        cfg["model"],
        trainable_lora=True,
        lora_cfg=train_cfg["lora"],
    )
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Check LoRA target modules.")
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["lr"]))

    rng = random.Random(int(cfg["seed"]))
    log_path = output / "logs" / "rl_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = 0.0
    baseline_ema = float(train_cfg["baseline_ema"])
    batch_size = int(train_cfg["batch_size"])
    n_agents = int(cfg["debate"]["agents"])
    use_curriculum = curriculum_enabled(cfg)

    warmup_steps = int(train_cfg.get("sft_warmup_steps", 0))
    if warmup_steps > 0:
        run_sft_warmup(
            cfg=cfg,
            output_dir=output,
            model=model,
            optimizer=optimizer,
            trainable=trainable,
            tasks=tasks,
            rng=rng,
            steps=warmup_steps,
        )

    with open(log_path, "w", encoding="utf-8") as log_f:
        for step in trange(1, int(train_cfg["steps"]) + 1, desc="rl-train"):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            batch_rewards: list[float] = []
            batch_events: list[dict[str, Any]] = []

            for _ in range(batch_size):
                task = rng.choice(tasks)
                if use_curriculum:
                    own_initial = oracle_private_answer(task)
                    peers = wrong_majority_peers(task, n_agents - 1, rng)
                else:
                    own_initial = model.generate(
                        private_answer_messages(task, agent_id=0),
                        max_new_tokens=int(cfg["debate"]["max_new_tokens"]),
                        temperature=float(cfg["debate"]["temperature"]),
                        top_p=float(cfg["debate"]["top_p"]),
                    )
                    peers = [
                        model.generate(
                            private_answer_messages(task, agent_id=i),
                            max_new_tokens=int(cfg["debate"]["max_new_tokens"]),
                            temperature=float(cfg["debate"]["temperature"]),
                            top_p=float(cfg["debate"]["top_p"]),
                        )
                        for i in range(1, n_agents)
                    ]
                final_messages = rl_final_messages(task, own_initial, peers)
                final_prompt = model.format_messages(final_messages)
                final_response = model.generate(
                    final_prompt,
                    max_new_tokens=int(cfg["debate"]["max_new_tokens"]),
                    temperature=float(cfg["debate"]["temperature"]),
                    top_p=float(cfg["debate"]["top_p"]),
                )

                reward = compute_reward(task, final_response, peers, own_initial, cfg["reward"])
                logprob, n_tokens = model.sequence_logprob(final_prompt, final_response)
                advantage = reward.total - baseline
                loss = -float(advantage) * (logprob / max(1, n_tokens))
                batch_losses.append(loss)
                batch_rewards.append(float(reward.total))
                batch_events.append(
                    {
                        "step": step,
                        "task_id": task.id,
                        "gold": task.answer,
                        "own_initial": own_initial,
                        "peers": peers,
                        "final_response": final_response,
                        "reward": asdict(reward),
                        "advantage": float(advantage),
                        "tokens": n_tokens,
                        "wrong_majority_curriculum": use_curriculum,
                    }
                )

            total_loss = sum(batch_losses) / len(batch_losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg["grad_clip"]))
            optimizer.step()

            mean_reward = sum(batch_rewards) / len(batch_rewards)
            baseline = baseline_ema * baseline + (1.0 - baseline_ema) * mean_reward
            for event in batch_events:
                event["loss"] = float(total_loss.detach().cpu().item())
                event["mean_reward"] = float(mean_reward)
                event["baseline"] = float(baseline)
                log_f.write(json.dumps(event, sort_keys=True) + "\n")
            log_f.flush()

            if step % int(train_cfg["save_every"]) == 0:
                model.save_adapter(output / "checkpoints" / f"step_{step:06d}")

    final_path = output / "checkpoints" / "final_adapter"
    model.save_adapter(final_path)
    return final_path


def run_sft_warmup(
    cfg: dict[str, Any],
    output_dir: Path,
    model: HFGenerator,
    optimizer: Any,
    trainable: list[Any],
    tasks: list[Any],
    rng: random.Random,
    steps: int,
) -> None:
    import torch

    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    n_peers = max(0, int(cfg["debate"]["agents"]) - 1)
    log_path = output_dir / "logs" / "sft_warmup.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_f:
        for step in trange(1, steps + 1, desc="sft-warmup"):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            events: list[dict[str, Any]] = []
            for _ in range(batch_size):
                task = rng.choice(tasks)
                messages, completion = rng.choice(supervised_examples(task, n_peers, rng))
                prompt = model.format_messages(messages)
                logprob, n_tokens = model.sequence_logprob(prompt, completion)
                loss = -(logprob / max(1, n_tokens))
                losses.append(loss)
                events.append(
                    {
                        "step": step,
                        "task_id": task.id,
                        "gold": task.answer,
                        "completion": completion,
                        "tokens": n_tokens,
                    }
                )

            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(train_cfg["grad_clip"]))
            optimizer.step()

            for event in events:
                event["loss"] = float(total_loss.detach().cpu().item())
                log_f.write(json.dumps(event, sort_keys=True) + "\n")
            log_f.flush()
