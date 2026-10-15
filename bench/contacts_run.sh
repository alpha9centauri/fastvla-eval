#!/usr/bin/env bash
# usage: contacts_run.sh <split|parity> <contacts 0|1> <outtag>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  MODE="$1" CONTACTS="$2" \
  ACTIONS_NPZ="$ROOT/audit/t1_out/actions_I0_ctrl.npz" \
  OUT="$ROOT/bench/out/m3_contacts_$3" N_CALLS="${N_CALLS:-1000}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/labeler/bench_contacts.py
