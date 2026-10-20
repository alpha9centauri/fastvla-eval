#!/bin/bash
# 7a final: planner plans grasp->place, dgcore does only the release. Certify at 1 cm.
# Author: Varun Chandra. GPU 1 only.
set -o pipefail
F=/data_new/vchandr2/fastvla-eval
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
cd "$F" || exit 1
env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
    MODE=full N=0 RES=256 SEED=0 FLOOR_M=0.01 TAGOUT=_4c \
    PLANNER_TERMINAL=place PLANNER_INFLATE_M=0.003 PLANNER_INFLATE_SUPPORT_M=0.0 \
    PLANNER_PAYLOAD_SPHERE_R=0.012 PLANNER_PAYLOAD_SPHERES=384 \
    PLANNER_CERTIFY_FLOOR_M=0.01 \
    $PY -u $F/fastvla/planner/step4b_run.py
rc=$?
[ $rc -ne 0 ] && { echo "[ALERT] 7a final FAILED|exit $rc"; exit $rc; }
echo "[ALERT] 7a final done|129 scenes, plan-to-place, certify 1 cm"
