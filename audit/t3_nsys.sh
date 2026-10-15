#!/usr/bin/env bash
# Table 3: Nsight Systems trace of ONE rollout, then the stats reports the idle-gap
# summary is computed from. perf_event_paranoid is 3 on this box, so CPU sampling and
# backtraces are off; CUDA/OSRT tracing works. ncu is used only for GPU kernel metrics.
set -euo pipefail
A="$(cd "$(dirname "$0")" && pwd)"
cd "$A"
OUT="$A/traces/t3_rollout"
env CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  PYTHONPATH=/data_new/vchandr2/rollout_stubs \
  MODE=timers OUT="$A/arrays/t2_nsys" \
  OUTDIR="$A/t3_out" STEP=17500 N_INIT=10 SMOKE_INITS=1 \
  PROBE=0 WRITE_BASELINE=0 STEER_K=0 INSTR_MODE=I0 OFFSETS=0 ARMS= USE_FEASIBLE=0 \
  ROWS_JSON=rows_t3.json CLAMPS_JSON=clamps_t3.json \
  TRAJ_NPZ=traj_t3.npz ACT_NPZ=actions_t3.npz \
  /usr/local/cuda/bin/nsys profile \
    --output "$OUT" --force-overwrite true \
    --trace=cuda,cudnn,cublas,osrt \
    --sample=none --cpuctxsw=none --backtrace=none \
    --cuda-memory-usage true \
    /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u t2_breakdown.py
echo "--- nsys stats ---"
for r in cuda_gpu_kern_sum cuda_gpu_trace cuda_api_sum; do
  /usr/local/cuda/bin/nsys stats --report "$r" --format csv \
    --output "$A/traces/t3_$r" "$OUT.nsys-rep" || echo "report $r unavailable"
done
ls -la "$A/traces/"
