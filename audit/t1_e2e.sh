#!/usr/bin/env bash
# Table 1: end-to-end wall-clock of one full 60-start evaluation of checkpoint 17500
# on GPU 1 with the CURRENT driver, unmodified. This is the ruler for M1.
#
# The command is the handoff's reference command verbatim, with two documented changes:
#   1. OUTDIR -> audit/t1_out, so no reference output file is touched.
#   2. MUJOCO_EGL_DEVICE_ID=1 added. The pasted reference command omits it, but EGL
#      device enumeration ignores CUDA_VISIBLE_DEVICES, so without it MuJoCo renders
#      on EGL device 0 = GPU 0, which belongs to another user. The handoff's own hard
#      rule ("MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 for rendering") requires it.
#      Acceptance is still 29/60 collisions and 25 or 26 completions.
set -euo pipefail
D=/data_new/vchandr2/vla-safety-interp/src/displacement
cd "$D"
exec env \
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  OUTDIR=/data_new/vchandr2/fastvla-eval/audit/t1_out \
  STEP=17500 N_INIT=10 PROBE=0 WRITE_BASELINE=0 STEER_K=0 \
  INSTR_MODE=I0 OFFSETS=0 ARMS= USE_FEASIBLE=0 \
  ROWS_JSON=rows_I0_ctrl.json CLAMPS_JSON=clamps_I0_ctrl.json \
  TRAJ_NPZ=traj_I0_ctrl.npz ACT_NPZ=actions_I0_ctrl.npz \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u rollout_openvla_disp.py
