#!/usr/bin/env bash
# M2 step 3: C++ vs Python parity on the 1000 saved states from audit Table 4.
set -euo pipefail
ROOT=/data_new/vchandr2/fastvla-eval
cd "$ROOT"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  ACTIONS_NPZ="$ROOT/audit/t1_out/actions_I0_ctrl.npz" \
  OUT="$ROOT/bench/out/m2_parity_real" \
  N_CALLS="${N_CALLS:-1000}" DISTMAX=0.5 CLEARANCE_EVERY=25 \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u fastvla/labeler/bench_parity.py
