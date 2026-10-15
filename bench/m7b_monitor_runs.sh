#!/usr/bin/env bash
# M7b steps 3-5: the three GPU runs the monitor needs, in order, one at a time.
#
#   m7b_log_ctrl60    MONITOR=log  on the 60-start acceptance set  -> step 3 cost and
#                     throughput, and step 5's parity proof (same run, same 60 starts:
#                     test_unshifted and ctrl60 are byte-identical manifests)
#   m7b_stop_ctrl60   MONITOR=stop on the same 60                 -> step 4
#   m7b_stop_intr5    MONITOR=stop on the 48 intrusion-5 starts   -> step 4
#
# N=8, B=1 throughout, to compare against M4_T1's 544.8 rollouts/hour.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${MONITOR_PROBE:-$ROOT/bench/out/m7b_probe.npz}"
[ -f "$PROBE" ] || { echo "no probe at $PROBE -- run fit_eval.py first" >&2; exit 1; }
cd "$ROOT/audit"
for cfg in "ctrl60 60 m7b_log_ctrl60 log" "ctrl60 60 m7b_stop_ctrl60 stop" \
           "intr5 48 m7b_stop_intr5 stop"; do
  set -- $cfg
  echo "=== $3: MANIFEST=$1 MONITOR=$4 N=8 B=1"
  rm -f "$ROOT/audit/DONE" "$ROOT/audit/FAILED"
  MONITOR="$4" MONITOR_PROBE="$PROBE" RUN_ID="$3" \
    DISPLACEMENT_CMD="bash $ROOT/bench/m3a_run.sh $1 8 1 0" \
    RUN_LOG_NAME="M7B_$3" DISPLACEMENT_TOTAL="$2" HEARTBEAT_SEC=1800 \
    /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u "$ROOT/audit/run_with_alert.py" \
      --no-wandb > "$ROOT/audit/logs/m7b_$3.log" 2>&1
done
echo "ALL M7B MONITOR RUNS DONE"
