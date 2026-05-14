#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/vast_0_5b.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${2:-runs/${STAMP}}"

truth-debate run --config "${CONFIG}" --output "${OUT}"
echo "Run complete: ${OUT}"
