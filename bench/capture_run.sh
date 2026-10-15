#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  OUT="$ROOT/bench/out/m4_requests.npz" N_PER_TASK="${N_PER_TASK:-2}" MAX_STEPS="${MAX_STEPS:-120}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u bench/capture_requests.py
