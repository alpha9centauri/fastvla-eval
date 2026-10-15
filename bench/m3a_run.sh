#!/usr/bin/env bash
# usage: m3a_run.sh <manifest> <N> <B> <D_ms> [extra env already exported]
set -euo pipefail
ROOT=/data_new/vchandr2/fastvla-eval
cd "$ROOT"
exec env MANIFEST="$1" N_WORKERS="$2" MAX_BATCH="$3" BATCH_DELAY_MS="$4" \
  RUN_ROOT="$ROOT/bench/runs" \
  PARITY="${PARITY:-0}" LABELER="${LABELER:-cpp}" CONTACTS="${CONTACTS:-1}" \
  MAX_STEPS="${MAX_STEPS:-300}" RUN_ID="${RUN_ID:-}" VIDEO_KEYS="${VIDEO_KEYS:-}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/eval/runner.py
