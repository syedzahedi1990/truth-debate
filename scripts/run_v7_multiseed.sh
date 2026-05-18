#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/v7_gsm8k_step5_seed1618_1_5b.yaml"
  "configs/v7_gsm8k_step5_seed2718_1_5b.yaml"
  "configs/v7_gsm8k_step5_seed3141_1_5b.yaml"
)

for config in "${configs[@]}"; do
  stem="$(basename "${config}" .yaml)"
  output="runs/${stem}"
  echo "=== Running ${config} -> ${output} ==="
  truth-debate run --config "${config}" --output "${output}"
  truth-debate rescore --source "${output}" --output "${output}_rescored"
done
