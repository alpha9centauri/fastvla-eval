#!/usr/bin/env bash
set -euo pipefail
A=/data_new/vchandr2/fastvla-eval/audit
cd "$A"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  ACTIONS_NPZ="$A/t1_out/actions_I0_ctrl.npz" OUT="$A/arrays/t4_labeler" \
  N_CALLS="${N_CALLS:-1000}" CLEARANCE_EVERY=25 DISTMAX_LIST=0.5,0.1,0.05 \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u t4_labeler.py
