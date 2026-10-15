#!/usr/bin/env bash
# M5 step 6: run the 60-start benchmark with the whole harness INSIDE the container.
#
# One `apptainer exec` runs the runner, and the runner spawns the server and the N workers
# as child processes inside that same container instance -- so every process in the
# measurement is containerised, not just one of them.
#
# Which image, and why not the CI-built one: CI builds deploy/Dockerfile.cpu, which is
# CPU-only by necessity (the GPU image does not fit a hosted runner). It therefore cannot
# run this benchmark. The image used here is the CUDA 12.1 container already on the box,
# with the libero-vla env bound and reached through PYTHONPATH -- the same pattern
# CONTAINERS.md section 8 requires and that M3b's Triton run used. Pulling the private
# ghcr package needs a credential this box does not have.
#
# Cold start measured here is "container start -> server reports READY", which is what the
# runner already times; image pull is excluded because the SIF is already local.
set -euo pipefail
export PATH=$HOME/apptainer/bin:$PATH
ROOT=/data_new/vchandr2/fastvla-eval
SIF=${SIF:-$HOME/images/tritonserver_23.07-py3.sif}
ENVROOT=/data_new/vchandr2/miniconda3/envs/libero-vla
SP=$ENVROOT/lib/python3.10/site-packages
PYPATH="$SP:/data_new/vchandr2/rollout_stubs:/data_new/vchandr2:/data_new/vchandr2/openvla-oft:/data_new/vchandr2/vla-safety-interp/src/displacement:$ROOT:$ROOT/fastvla/labeler"

N=${N_WORKERS:-8}
B=${MAX_BATCH:-1}
MAN=${MANIFEST:-ctrl60}
RID=${RUN_ID:-incontainer}

echo "[INCONTAINER] image: $SIF"
# The container images lack libX11/libXext/libGL, which robosuite and cv2 both need.
# bench/make_container_libs.sh builds the curated directory; only graphics libraries, and
# APPENDED to LD_LIBRARY_PATH so the container's own libraries keep winning.
[ -d "$ROOT/bench/triton_libs" ] || bash "$ROOT/bench/make_container_libs.sh"

exec env CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
  --bind /data_new/vchandr2:/data_new/vchandr2 \
  --env PYTHONPATH="$PYPATH" \
  --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  --env MUJOCO_GL=egl --env MUJOCO_EGL_DEVICE_ID=1 \
  --env MANIFEST="$MAN" --env N_WORKERS="$N" --env MAX_BATCH="$B" \
  --env BATCH_DELAY_MS=0 --env RUN_ID="$RID" \
  --env RUN_ROOT="$ROOT/bench/runs" \
  --env LABELER=cpp --env CONTACTS=1 --env LABELER_LEVEL=batch --env LABELER_THREADS=1 \
  --env POLICY_TRANSPORT=zmq --env METRICS_PORT=9401 \
  "$SIF" sh -c 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}'"$ROOT"'/bench/triton_libs"; \
    exec '"$ENVROOT"'/bin/python -u '"$ROOT"'/fastvla/eval/runner.py' 
