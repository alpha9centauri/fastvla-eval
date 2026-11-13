#!/bin/bash
# The nine: the eight October near-plate oa1 scenes that planned but did not certify, and the one
# planner_v2 position that planned but did not certify. ONE PER PROCESS (stale attachment).
# GPU 1 only, strictly sequential. Author: Varun Chandra.
set -o pipefail
V=/data_new/vchandr2/vla_old_research/fastvla-eval/fastvla/planner/v2
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
for t in 0:12 1:13 2:6 4:8 4:10 5:7 8:11 9:4 v2single; do
  echo "=== scene $t ==="
  env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
      MODE=full RES=256 SEED=0 FLOOR_M=0.02 SCENE="$t" \
      PLANNER_TERMINAL=place PLANNER_INFLATE_M=0.003 PLANNER_INFLATE_SUPPORT_M=0.0 \
      PLANNER_PAYLOAD_SPHERE_R=0.012 PLANNER_PAYLOAD_SPHERES=384 \
      PLANNER_CERTIFY_FLOOR_M=0.01 \
      timeout 1800 $PY -u $V/labelled_rerun.py 2>&1 \
    | grep -E "^\[LAB\]|^      |LABELLED_ONE_DONE|Traceback|Error" \
    || echo "  scene $t: process failed or timed out"
done
echo "LABELLED_ALL_DONE"
