#!/bin/bash
# Re-run chosen scenes ONE PER PROCESS, fresh MotionGen each, same settings as the 7a final run.
# Why: the final run reported ik_place for scenes whose place pose is 21-22 cm clear, and the same
# pose returns IK=True through the router's own call in a fresh process. A long-lived router with
# per-scene attach/detach is the suspect, so each scene gets its own process and nothing can carry.
# Author: Varun Chandra. GPU 1 only, strictly sequential.
set -o pipefail
F=/data_new/vchandr2/fastvla-eval
PY=/data_new/vchandr2/miniconda3/envs/planner/bin/python
TG="$1"
[ -z "$TG" ] && { echo "usage: $0 shard:attempt,shard:attempt,..."; exit 2; }
cd "$F" || exit 1
IFS=',' read -ra ONE <<< "$TG"
for t in "${ONE[@]}"; do
  echo "=== scene $t ==="
  env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
      PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=/data_new/vchandr2/rollout_stubs \
      MODE=full RES=256 SEED=0 FLOOR_M=0.01 TAGOUT="_one_${t//:/_}" TARGETS="$t" \
      PLANNER_TERMINAL=place PLANNER_INFLATE_M=0.003 PLANNER_INFLATE_SUPPORT_M=0.0 \
      PLANNER_PAYLOAD_SPHERE_R=0.012 PLANNER_PAYLOAD_SPHERES=384 \
      PLANNER_CERTIFY_FLOOR_M=0.01 \
      timeout 1200 $PY -u $F/fastvla/planner/step4b_run.py 2>&1 \
    | grep -E "^  s[0-9]+a[0-9]+|returned a path|ACCEPTED at|planner_failure|executor|STEP4B_FULL_DONE"
done
echo "ONE_PER_PROCESS_DONE"
