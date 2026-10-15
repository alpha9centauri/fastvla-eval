#!/usr/bin/env bash
# M7b follow-up: MONITOR=stop at the chosen operating point (the knee of the recall /
# false-alarm curve, with a 2-forward debounce). Threshold and debounce come from the
# probe npz, so these runs use exactly what EVAL.md reports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="${MONITOR_PROBE:-$ROOT/bench/out/m7b_probe.npz}"
[ -f "$PROBE" ] || { echo "no probe at $PROBE" >&2; exit 1; }
cd "$ROOT/audit"
for cfg in "ctrl60 60 m7b_stopknee_ctrl60" "intr5 48 m7b_stopknee_intr5"; do
  set -- $cfg
  echo "=== $3: MANIFEST=$1 MONITOR=stop N=8 B=1 (threshold and debounce from the probe)"
  rm -f "$ROOT/audit/DONE" "$ROOT/audit/FAILED"
  MONITOR=stop MONITOR_PROBE="$PROBE" RUN_ID="$3" \
    DISPLACEMENT_CMD="bash $ROOT/bench/m3a_run.sh $1 8 1 0" \
    RUN_LOG_NAME="M7B_$3" DISPLACEMENT_TOTAL="$2" HEARTBEAT_SEC=1800 \
    /data_new/vchandr2/miniconda3/envs/libero-vla/bin/python -u "$ROOT/audit/run_with_alert.py" \
      --no-wandb > "$ROOT/audit/logs/m7b_$3.log" 2>&1
done
echo "ALL M7B KNEE STOP RUNS DONE"
