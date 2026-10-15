#!/usr/bin/env bash
# M2 step 6: the 60-start Table 1 run through the selected labeler. Identical to
# audit/t1_e2e.sh except for OUTDIR and the labeler flag.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D=/data_new/vchandr2/vla-safety-interp/src/displacement
TAG="${4:-cpp}"
cd "$D"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  LABELER="${1:-cpp}" LABELER_LEVEL="${2:-batch}" LABELER_THREADS="${3:-1}" \
  TARGET="$D/rollout_openvla_disp.py" \
  OUTDIR="$ROOT/bench/out/t1_$TAG" \
  STEP=17500 N_INIT=10 PROBE=0 WRITE_BASELINE=0 STEER_K=0 \
  INSTR_MODE=I0 OFFSETS=0 ARMS= USE_FEASIBLE=0 \
  ROWS_JSON=rows_I0_ctrl.json CLAMPS_JSON=clamps_I0_ctrl.json \
  TRAJ_NPZ=traj_I0_ctrl.npz ACT_NPZ=actions_I0_ctrl.npz \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u "$ROOT/fastvla/run_with_labeler.py"
