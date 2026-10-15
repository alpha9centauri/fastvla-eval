#!/usr/bin/env bash
# usage: call_run.sh <python|cpp> <gjk|batch> <threads> <outtag>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  LABELER="${1}" LABELER_LEVEL="${2}" LABELER_THREADS="${3}" \
  ACTIONS_NPZ="$ROOT/audit/t1_out/actions_I0_ctrl.npz" \
  OUT="$ROOT/bench/out/m2_call_${4}" \
  N_CALLS="${N_CALLS:-1000}" DISTMAX=0.5 CLEARANCE_EVERY=25 \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/labeler/bench_call.py
