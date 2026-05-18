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

## V2 Smoke Run

After the first diagnostic run, use the v2 smoke config before spending money on a long A100 run:

```bash
truth-debate run --config configs/v2_smoke.yaml --output runs/v2_smoke
truth-debate rescore --source runs/v2_smoke --output runs/v2_smoke_rescored
```

The v2 config changes the experiment in three ways:

- all prompts ask for one JSON object with `answer` and `confidence`;
- supervised warmup teaches the output format before RL;
- RL uses an oracle wrong-majority curriculum so the anti-conformity reward is active.

Do not scale until the smoke run has parse failure below about 5%, anti-conformity reward activation well above the first run's 0.17%, and no post-training accuracy collapse.

## V3 Computation SFT And Benchmark Run

V3 targets the failure mode found in V2: the model learned valid JSON but still computed private answers poorly. It adds exact computation rationales during SFT and mixes oracle-private and model-private RL curriculum cases.

Run the controlled debugging benchmark first:

```bash
truth-debate run --config configs/v3_synthetic_smoke.yaml --output runs/v3_synthetic_smoke
truth-debate rescore --source runs/v3_synthetic_smoke --output runs/v3_synthetic_smoke_rescored
```

Then run the standard-benchmark smoke on GSM8K:

```bash
truth-debate run --config configs/v3_gsm8k_smoke.yaml --output runs/v3_gsm8k_smoke
truth-debate rescore --source runs/v3_gsm8k_smoke --output runs/v3_gsm8k_smoke_rescored
```

The synthetic benchmark is for controlled debugging. GSM8K is the standard benchmark sanity check; the smoke config uses a subset before a full benchmark run.

## V4 Stable GSM8K Runs

V3 showed that GSM8K training can collapse during RL even when the synthetic benchmark improves. V4 separates the diagnosis into a supervised-format run and a short anchored-RL run.

First isolate whether supervised warmup alone improves GSM8K compliance:

```bash
truth-debate run --config configs/v4_gsm8k_sft_only.yaml --output runs/v4_gsm8k_sft_only
truth-debate rescore --source runs/v4_gsm8k_sft_only --output runs/v4_gsm8k_sft_only_rescored
```

Then run the conservative RL variant:

```bash
truth-debate run --config configs/v4_gsm8k_stable_rl.yaml --output runs/v4_gsm8k_stable_rl
truth-debate rescore --source runs/v4_gsm8k_stable_rl --output runs/v4_gsm8k_stable_rl_rescored
```

The v4 RL config saves `post_sft_adapter` and checkpoints every 10 RL steps. To evaluate a specific checkpoint:

```bash
truth-debate eval --config configs/v4_gsm8k_stable_rl.yaml --output runs/v4_gsm8k_stable_rl --label step20 --adapter-path runs/v4_gsm8k_stable_rl/checkpoints/step_000020
truth-debate rescore --source runs/v4_gsm8k_stable_rl --output runs/v4_gsm8k_stable_rl_rescored_step20
```

For GSM8K reports, strict JSON accuracy remains the primary protocol-compliance metric. Reports also include `standard numeric accuracy`, which extracts the final numeric answer in the usual benchmark style so format failure can be separated from math failure.

## V5 Stronger GSM8K Runs

The v4 SFT-only run showed that the 0.5B model learned JSON compliance but lost GSM8K numeric accuracy. V5 switches to `Qwen/Qwen2.5-1.5B-Instruct`, uses much lighter SFT, and lets RL rewards use standard numeric answer parsing while reports still separate strict JSON accuracy from benchmark-style numeric accuracy.

Run the lighter SFT-only diagnosis first:

```bash
truth-debate run --config configs/v5_gsm8k_light_sft_1_5b.yaml --output runs/v5_gsm8k_light_sft_1_5b
truth-debate rescore --source runs/v5_gsm8k_light_sft_1_5b --output runs/v5_gsm8k_light_sft_1_5b_rescored
```

Only run anchored RL if the post-SFT standard numeric accuracy is at least close to the baseline numeric accuracy:

```bash
truth-debate run --config configs/v5_gsm8k_stable_rl_1_5b.yaml --output runs/v5_gsm8k_stable_rl_1_5b
truth-debate rescore --source runs/v5_gsm8k_stable_rl_1_5b --output runs/v5_gsm8k_stable_rl_1_5b_rescored
```

The v5 RL config checkpoints every 10 RL steps and uses `reward.answer_parse_mode: standard_numeric`, so the RL signal rewards correct GSM8K answers even if the output is not perfect JSON. The separate strict parse metrics still show whether the protocol-compliance story holds.

## V6 Hardened Anti-Conformity Runs

V5 showed that the 1.5B baseline is usable, but anti-conformity sometimes emits nested or truncated JSON. V6 hardens the anti-conformity prompts to require flat JSON with `answer` as the first key, and keeps RL off until the new baseline is inspected.

Run the baseline-only check first:

```bash
truth-debate run --config configs/v6_gsm8k_baseline_1_5b.yaml --output runs/v6_gsm8k_baseline_1_5b
truth-debate rescore --source runs/v6_gsm8k_baseline_1_5b --output runs/v6_gsm8k_baseline_1_5b_rescored
```

Then, only if the hardened baseline preserves the v5 vanilla-debate signal and reduces anti-conformity parse failures, run the no-SFT RL variant:

```bash
truth-debate run --config configs/v6_gsm8k_no_sft_rl_1_5b.yaml --output runs/v6_gsm8k_no_sft_rl_1_5b
truth-debate rescore --source runs/v6_gsm8k_no_sft_rl_1_5b --output runs/v6_gsm8k_no_sft_rl_1_5b_rescored
```

The no-SFT RL run starts from the base model rather than a supervised checkpoint. It uses `reward.answer_parse_mode: standard_numeric`, a small learning rate, advantage clipping, and checkpoints every 5 RL steps.

Useful subcommands:

```bash
truth-debate make-data --config configs/quick.yaml --output runs/debug
truth-debate eval --config configs/quick.yaml --output runs/debug
truth-debate eval --config configs/quick.yaml --output runs/debug --label checkpoint --adapter-path runs/debug/checkpoints/step_000010
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
