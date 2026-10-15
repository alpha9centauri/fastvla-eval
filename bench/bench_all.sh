#!/usr/bin/env bash
# `make bench`: reproduce every number in the README benchmark table.
#
# Stages, in dependency order. Each writes a JSON under bench/out/ and the table is
# rendered from those files by bench/render_table.py -- so a number in the README cannot
# drift from what was measured.
#
#   1  60-start evaluation, N=8 B=1          -> rollouts/hour, GPU util, acceptance
#                                              (also writes the saved actions the
#                                               labeler and forward benches replay)
#   2  labeler microbenchmark, python vs C++ -> ms per clearance call
#   3  forward microbenchmark                -> ms per forward, per phase
#   4  acceptance against the reference      -> collisions, completions, flips
#
# Runtime on one RTX A5000: about 25 minutes. BENCH_QUICK=1 skips stages 2 and 3, which
# are the slow replays, and reproduces the end-to-end rows only (about 8 minutes).
#
# Requires: GPU 1 free, the libero-vla env, and the research tree for the replay stages.
# bench/make_container_libs.sh is needed only for the in-container variant.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PY:-/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python}
A="$ROOT/audit"
STAMP=$(date +%H%M%S)
mkdir -p bench/out

run_stage () {   # name, run_log_name, command
  local name=$1 log=$2; shift 2
  echo "=== bench stage: $name ==="
  rm -f "$A/DONE" "$A/FAILED"
  ( cd "$A" && DISPLACEMENT_CMD="$*" RUN_LOG_NAME="$log" DISPLACEMENT_TOTAL=60 \
      HEARTBEAT_SEC=1800 "$PY" -u run_with_alert.py --no-wandb ) || {
        echo "stage $name FAILED"; return 1; }
}

echo "[BENCH] pre-flight (same check every GPU job runs; see CONTAINERS.md section 8)"
CUDA_VISIBLE_DEVICES=1 "$PY" fastvla/server/preflight_gpu.py

run_stage "60-start, N=8 B=1" "BENCH_T1_$STAMP" \
  "bash $ROOT/bench/sweep.sh ctrl60 8 1 0 bench_t1_$STAMP"
"$PY" bench/accept.py "bench_t1_$STAMP" bench_t1 || true

if [ "${BENCH_QUICK:-0}" != "1" ]; then
  run_stage "labeler, python" "BENCH_LAB_PY_$STAMP" \
    "bash $ROOT/bench/call_run.sh python batch 1 bench_lab_python"
  run_stage "labeler, C++" "BENCH_LAB_CPP_$STAMP" \
    "bash $ROOT/bench/call_run.sh cpp batch 1 bench_lab_cpp"
  run_stage "forward microbenchmark" "BENCH_FWD_$STAMP" \
    "bash $ROOT/bench/fwd_run.sh time baseline bench_fwd 50"
fi

echo "=== rendering the table ==="
"$PY" bench/render_table.py
