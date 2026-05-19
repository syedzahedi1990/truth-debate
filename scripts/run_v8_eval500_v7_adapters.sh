#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/v8_gsm8k_eval500_seed1618_1_5b.yaml"
  "configs/v8_gsm8k_eval500_seed2718_1_5b.yaml"
  "configs/v8_gsm8k_eval500_seed3141_1_5b.yaml"
)

adapters=(
  "runs/v7_gsm8k_step5_seed1618_1_5b/checkpoints/final_adapter"
  "runs/v7_gsm8k_step5_seed2718_1_5b/checkpoints/final_adapter"
  "runs/v7_gsm8k_step5_seed3141_1_5b/checkpoints/final_adapter"
)

for idx in "${!configs[@]}"; do
  config="${configs[$idx]}"
  adapter="${adapters[$idx]}"
  stem="$(basename "${config}" .yaml)"
  output="runs/${stem}"

  if [[ ! -d "${adapter}" ]]; then
    echo "Missing adapter: ${adapter}" >&2
    echo "Run bash scripts/run_v7_multiseed.sh first, or restore the v7 run directories." >&2
    exit 1
  fi

  echo "=== Evaluating ${config} with adapter ${adapter} ==="
  truth-debate eval --config "${config}" --output "${output}" --label baseline
  truth-debate eval --config "${config}" --output "${output}" --label trained --adapter-path "${adapter}"
  truth-debate rescore --source "${output}" --output "${output}_rescored"
done
