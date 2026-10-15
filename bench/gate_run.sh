#!/usr/bin/env bash
# M2 step 6: the 12-minute gate (datagen/curve_gate.py) through the selected labeler.
# Reference: 10/10 collisions, 0.00 cm min clearance on every start, 3/10 completions on
# starts {4, 6, 9}; exit 0 on match, 3 otherwise.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D=/data_new/vchandr2/vla-safety-interp/src/displacement
CKPT=/data_new/vchandr2/runs/openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--libero_safety--warmstart-combined--headinit--17500_chkpt
cd "$D"
exec env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  CKPT="$CKPT" \
  LABELER="${1:-cpp}" LABELER_LEVEL="${2:-batch}" LABELER_THREADS="${3:-1}" \
  TARGET="$D/datagen/curve_gate.py" \
  REDIRECT_WRITE="curve_gate.json:$ROOT/bench/out/m2_curve_gate_${1:-cpp}.json" \
  RECORD_ACTIONS="${RECORD_ACTIONS:-}" \
  /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u "$ROOT/fastvla/run_with_labeler.py"
