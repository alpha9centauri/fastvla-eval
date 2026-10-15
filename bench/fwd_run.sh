#!/usr/bin/env bash
# usage: fwd_run.sh <mode> <variant> <outtag> [n_req]
set -euo pipefail
ROOT=/data_new/vchandr2/fastvla-eval
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  MODE="$1" VARIANT="$2" OUT="$ROOT/bench/out/m4_fwd_$3" N_REQ="${4:-50}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/server/bench_forward.py
