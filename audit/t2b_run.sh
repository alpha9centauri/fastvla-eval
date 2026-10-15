#!/usr/bin/env bash
set -euo pipefail
A=/data_new/vchandr2/fastvla-eval/audit
cd "$A"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  OUT="$A/arrays/t2b_render_split" N_STEPS=100 \
  OUTDIR="$A/t2b_out" STEP=17500 N_INIT=1 SMOKE_INITS=1 MAX_STEPS=100 \
  PROBE=0 WRITE_BASELINE=0 STEER_K=0 INSTR_MODE=I0 OFFSETS=0 ARMS= USE_FEASIBLE=0 \
  ROWS_JSON=rows_t2b.json CLAMPS_JSON=clamps_t2b.json \
  TRAJ_NPZ=traj_t2b.npz ACT_NPZ=actions_t2b.npz \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u t2b_render_split.py
