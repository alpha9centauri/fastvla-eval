#!/usr/bin/env bash
# Start Triton inside the Apptainer container, serving fastvla/server/triton/models.
#
# Invocation rules from fastvla/server/CONTAINERS.md section 8: plain --nv with
# CUDA_VISIBLE_DEVICES=1, never --contain, and preflight_gpu.py first. The libero-vla env
# is bound and reached through PYTHONPATH so Triton's Python backend (container CPython
# 3.10.6, same cp310 ABI as the env's 3.10.20) imports the identical stack the ZeroMQ
# server uses -- which is what makes the comparison about serving layers, not stacks.
#
# Non-default ports, since someone else may run Triton on this box.
set -euo pipefail
export PATH=$HOME/apptainer/bin:$PATH
ROOT=/data_new/vchandr2/fastvla-eval
SIF=${SIF:-$HOME/images/tritonserver_23.07-py3.sif}
ENVROOT=/data_new/vchandr2/miniconda3/envs/libero-vla
BASE="openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_safety--warmstart-combined--headinit"
CKPT=${CKPT:-/data_new/vchandr2/runs/$BASE--17500_chkpt}
SP=$ENVROOT/lib/python3.10/site-packages
HTTP=${TRITON_HTTP_PORT:-8200}
GRPC=${TRITON_GRPC_PORT:-8201}
METRICS=${TRITON_METRICS_PORT:-8202}

PYPATH="$SP:/data_new/vchandr2/rollout_stubs:/data_new/vchandr2:/data_new/vchandr2/openvla-oft:/data_new/vchandr2/vla-safety-interp/src/displacement:$ROOT"

echo "[TRITON] pre-flight"
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
  --bind /data_new/vchandr2:/data_new/vchandr2 \
  --env PYTHONPATH="$PYPATH" \
  "$SIF" "$ENVROOT/bin/python" "$ROOT/fastvla/server/preflight_gpu.py"

echo "[TRITON] starting tritonserver (http $HTTP, grpc $GRPC, metrics $METRICS)"
# Write our own pid so bench/triton_stop.sh can stop this by pid. Killing by pgrep pattern
# has twice matched the caller's own command line and killed the wrong process.
echo $$ > "$ROOT/bench/out/triton_serve.pid"
# The Triton image has no libX11/libGL, which robosuite pulls in through libero. Only the
# GRAPHICS libraries are exposed, as a curated directory of symlinks (bench/triton_libs).
#
# Putting the env's whole lib/ on LD_LIBRARY_PATH does NOT work, even appended: it shadows
# the container's libtinfo and libstdc++, and numpy's C extension then fails to load with
# the misleading "you should not try to import numpy from its source directory". The
# giveaway was bash itself warning about the env's libtinfo.so.6.
exec env CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
  --bind /data_new/vchandr2:/data_new/vchandr2 \
  --env PYTHONPATH="$PYPATH" \
  --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  --env CKPT="$CKPT" \
  "$SIF" sh -c 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}'"$ROOT"'/bench/triton_libs"; \
    exec tritonserver \
      --model-repository='"$ROOT"'/fastvla/server/triton/models \
      --http-port='"$HTTP"' --grpc-port='"$GRPC"' --metrics-port='"$METRICS"' \
      --log-verbose='"${TRITON_VERBOSE:-0}"' \
      --exit-on-error=true'
