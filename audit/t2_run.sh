#!/usr/bin/env bash
# Table 2 driver. usage: t2_run.sh <timers|torch|cprofile> <n_rollouts>
set -euo pipefail
MODE_IN="${1:?mode}"; NROLL="${2:?n rollouts}"
A=/data_new/vchandr2/fastvla-eval/audit
cd "$A"
exec env \
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  MODE="$MODE_IN" OUT="$A/arrays/t2_${OUTTAG:-$MODE_IN}" COUNT_PAIRS="${COUNT_PAIRS:-1}" LABELER="${LABELER:-python}" LABELER_LEVEL="${LABELER_LEVEL:-batch}" LABELER_THREADS="${LABELER_THREADS:-1}" \
  OUTDIR="$A/t2_out" STEP=17500 N_INIT="${N_INIT:-10}" SMOKE_INITS="$NROLL" \
  PROBE=0 WRITE_BASELINE=0 STEER_K=0 INSTR_MODE=I0 OFFSETS=0 ARMS= USE_FEASIBLE=0 \
  ROWS_JSON="rows_t2_$MODE_IN.json" CLAMPS_JSON="clamps_t2_$MODE_IN.json" \
  TRAJ_NPZ="traj_t2_$MODE_IN.npz" ACT_NPZ="actions_t2_$MODE_IN.npz" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u t2_breakdown.py
