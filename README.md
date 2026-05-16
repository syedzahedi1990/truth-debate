# Truth-Seeking Debate

This repo runs an end-to-end experiment for **RL-trained truth-seeking multi-agent debate with anti-conformity protocols**.

The first benchmark is deliberately reproducible: synthetic arithmetic and precedence-trap tasks with exact labels. The pipeline can run fully unattended:

1. Generate train/eval tasks.
2. Evaluate a base model under `single`, `vanilla_debate`, and `anti_conformity` protocols.
3. Train a LoRA adapter with sequence-level policy gradient rewards for truthfulness and resistance to wrong consensus.
4. Re-evaluate the trained adapter.
5. Emit JSONL rollouts, metrics, and a Markdown report.

## Quick Local Smoke Test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest
python -m compileall src
```

The real run downloads a Hugging Face model and needs a GPU.

## Vast.ai / Hosted GPU Run

On a CUDA PyTorch image or a fresh Ubuntu GPU instance:

```bash
git clone <your-repo-url> truth-debate
cd truth-debate
bash scripts/bootstrap_vast.sh
source .venv/bin/activate
truth-debate preflight --config configs/vast_0_5b.yaml
bash scripts/run_end_to_end.sh configs/vast_0_5b.yaml
```

For gated models, set `HF_TOKEN` before running:

```bash
export HF_TOKEN=...
```

If the instance cannot reach Hugging Face, the run will fail before loading the model. Check:

```bash
curl -I https://huggingface.co
truth-debate preflight --config configs/vast_0_5b.yaml
```

If `curl` says the network is unreachable, the instance/container has no outbound route. Rent or restart a Vast.ai instance with public internet enabled, or pre-download the model elsewhere and point the config at the copied local directory:

```bash
truth-debate download-model --config configs/vast_0_5b.yaml --local-dir models/qwen2.5-0.5b-instruct
# then set model.name: models/qwen2.5-0.5b-instruct
```

Outputs are written to `runs/<timestamp>/`:

- `data/train.jsonl`, `data/eval.jsonl`
- `baseline_rollouts/*.jsonl`
- `trained_rollouts/*.jsonl`
- `checkpoints/final_adapter/`
- `metrics/*.json`
- `report.md`

## Main Command

```bash
truth-debate run --config configs/vast_0_5b.yaml --output runs/my_run
```

Useful subcommands:

```bash
truth-debate make-data --config configs/quick.yaml --output runs/debug
truth-debate eval --config configs/quick.yaml --output runs/debug
truth-debate train --config configs/quick.yaml --output runs/debug
truth-debate report --output runs/debug
truth-debate rescore --source runs/my_run --output runs/my_run_rescored
truth-debate rescore --source 20260514_110718.zip --output runs/20260514_110718_rescored
```

Use `rescore` on completed runs when parser behavior changes. It reads rollouts from either a run directory or zip archive and writes `rescored_metrics/` plus `rescored_report.md` without modifying the original run.

## Research Hypotheses

The automated experiment is built around three claims:

1. Vanilla debate can improve accuracy but can also amplify a wrong majority.
2. Anti-conformity structure should reduce wrong-consensus and correct-to-wrong flips.
3. RL can train a debater to preserve correct private beliefs and defect from wrong peer consensus without merely becoming contrarian.

## Protocols

`single`: one sampled answer.

`vanilla_debate`: each agent sees peer answers and updates. The final answer is parsed from the final agent messages and aggregated by majority.

`anti_conformity`: each agent makes a private commitment first. Updates require explicit error localization, a majority-risk note, and a minority report. Metrics track whether the protocol preserves minority answers and avoids wrong consensus.

## Reward

The RL reward combines:

- exact final-answer correctness,
- bonus for escaping a wrong peer majority,
- penalty for joining a wrong peer majority,
- penalty for flipping from a correct private answer to an incorrect final answer,
- optional small formatting bonus for parseable `ANSWER:` and `CONFIDENCE:` fields.

This is intentionally transparent so the paper can analyze reward hacking and ablate each component.

## Scaling Notes

Start with `Qwen/Qwen2.5-0.5B-Instruct` on a 24 GB GPU. Then scale:

- more train tasks,
- more RL steps,
- more debate agents and rounds,
- a larger model such as `Qwen/Qwen2.5-1.5B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`,
- harder datasets by adding loaders in `src/truth_debate/data.py`.

## Reproducibility

Every run stores the resolved config in `resolved_config.yaml`. JSONL rollouts include prompts, generated messages, parsed answers, rewards, and per-task metrics.
