#!/usr/bin/env bash
set -euo pipefail
ROOT=/data_new/vchandr2/fastvla-eval
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  OUT="$ROOT/bench/out/m3_probe_batching" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/server/probe_batching.py
