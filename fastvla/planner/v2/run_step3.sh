#!/bin/bash
# Step 3: plan and certify at every kept position, ONE POSITION PER PROCESS.
# A long-lived MotionGen carries the previous scene's attached payload (RESULTS_4B.md), so each
# position gets a fresh process. GPU 1 only, strictly sequential.
# Floors unchanged: certification 2 cm (FLOOR_M), planner terminal-gap pre-filter 1 cm.
# Author: Varun Chandra.
set -e
set -o pipefail
V=/data_new/vchandr2/vla_old_research/fastvla-eval/fastvla/planner/v2
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
N=$($PY -c "import json;print(len(json.load(open('$V/kept_positions.json'))))")
echo "=== step 3: $N kept positions, one process each ==="
for ((i=0; i<N; i++)); do
  env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
      MODE=full RES=256 SEED=0 FLOOR_M=0.02 POS_IDX=$i \
      PLANNER_TERMINAL=place PLANNER_INFLATE_M=0.003 PLANNER_INFLATE_SUPPORT_M=0.0 \
      PLANNER_PAYLOAD_SPHERE_R=0.012 PLANNER_PAYLOAD_SPHERES=384 \
      PLANNER_CERTIFY_FLOOR_M=0.01 \
      timeout 1200 $PY -u $V/step3_certify.py 2>&1 \
    | grep -E "^\[STEP3\]|returned a path|ACCEPTED at|planner_failure|executor|STEP3_ONE_DONE" \
    || echo "  position $i: process failed or timed out"
done
echo "STEP3_ALL_DONE"
