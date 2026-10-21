#!/bin/bash
# Step 4b full: the 129 step-4 scenes that returned a path, at the chosen inflation, through the
# real certifier (dry pass + 2 cm GJK gate + rendered replay gate).
# Author: Varun Chandra. GPU 1 only, one process, one warm cuRobo instance.
set -o pipefail
F=/data_new/vchandr2/fastvla-eval
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
cd "$F" || exit 1
env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
    MODE=full N=0 RES=256 SEED=0 FLOOR_M=0.01 \
    PLANNER_INFLATE_M=0.003 PLANNER_INFLATE_SUPPORT_M=0.0 \
    PLANNER_PAYLOAD_SPHERE_R=0.022 PLANNER_PAYLOAD_SPHERES=64 \
    $PY -u $F/fastvla/planner/step4b_run.py
rc=$?
[ $rc -ne 0 ] && { echo "[ALERT] step4b full FAILED|exit $rc"; exit $rc; }
echo "[ALERT] step4b full done|129 scenes at inflation 0.003 m"
