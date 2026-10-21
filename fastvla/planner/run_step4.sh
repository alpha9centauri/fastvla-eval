#!/bin/bash
# Step 4: the planner on the 2b smoke's eval-rule oa1 scenes, 10 shards x 20 attempts.
# Author: Varun Chandra. GPU 1 only, one process, one warm cuRobo instance.
set -o pipefail
F=/data_new/vchandr2/fastvla-eval
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
cd "$F" || exit 1
env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
    SHARDS=10 ATTEMPTS=20 SEED=0 RES=256 FLOOR_M=0.01 PLANNER=curobo \
    $PY -u $F/fastvla/planner/step4_measure.py
rc=$?
[ $rc -ne 0 ] && { echo "[ALERT] step4 FAILED|exit $rc"; exit $rc; }
echo "[ALERT] step4 done|200 eval-rule scenes measured"
