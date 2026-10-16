#!/usr/bin/env bash
# usage: m3a_run.sh <manifest> <N> <B> <D_ms> [extra env already exported]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env MANIFEST="$1" N_WORKERS="$2" MAX_BATCH="$3" BATCH_DELAY_MS="$4" \
  RUN_ROOT="$ROOT/bench/runs" \
  PARITY="${PARITY:-0}" LABELER="${LABELER:-cpp}" CONTACTS="${CONTACTS:-1}" \
  POLICY_TRANSPORT="${POLICY_TRANSPORT:-zmq}" TRITON_URL="${TRITON_URL:-localhost:8201}" \
  CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}" TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-0}" \
  SERVER_OPTS="${SERVER_OPTS:-}" \
  CAPTURE_ACTIVATIONS="${CAPTURE_ACTIVATIONS:-0}" \
  CKPT="${CKPT:-}" \
  MONITOR="${MONITOR:-off}" MONITOR_PROBE="${MONITOR_PROBE:-}" \
  MONITOR_THRESHOLD="${MONITOR_THRESHOLD:-}" MONITOR_DEBOUNCE="${MONITOR_DEBOUNCE:-}" \
  MAX_STEPS="${MAX_STEPS:-300}" RUN_ID="${RUN_ID:-}" VIDEO_KEYS="${VIDEO_KEYS:-}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/eval/runner.py
