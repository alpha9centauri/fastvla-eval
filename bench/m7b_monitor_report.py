"""Step 3's numbers: what the monitor costs per forward, and what it costs end to end.

Alarm counts are recomputed here from the per-forward `monitor_scores` array and the
probe's threshold rather than read from the rollout row: the row records held forwards
(`monitor_n_held_forwards`), not alarms, because the worker is never told the threshold.
The arrays are the authority; see EVAL.md section 10.

usage: m7b_monitor_report.py <out.json> <run_id> [run_id ...]
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "bench", "runs")


def one(run_id):
    rd = os.path.join(RUNS, run_id)
    with open(os.path.join(rd, "merged.json")) as fh:
        merged = json.load(fh)
    with open(os.path.join(rd, "config.json")) as fh:
        cfg = json.load(fh)
    sp = os.path.join(rd, "server_stats.json")
    st = {}
    if os.path.exists(sp):
        with open(sp) as fh:
            st = json.load(fh)
    summ = st.get("summary", {})
    mon = st.get("monitor_summary", {})
    thr = mon.get("threshold")
    n_cross = n_alarm_forwards = n_forwards = n_alarm_rollouts = 0
    for r in merged["rows"]:
        p = os.path.join(rd, "arrays", f"{r['key']}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if "monitor_scores" not in z.files:
            continue  # a run from before the monitor existed, used here as the baseline
        s = np.asarray(z["monitor_scores"], float)
        n_forwards += len(s)
        if thr is None or not len(s):
            continue
        n_cross += int((s >= thr).sum())
        # Crossings and alarms are different once a debounce is in play: the alarm is what
        # the monitor acts on. Runs made before the debounce existed have no such array,
        # and for them the two are the same thing.
        if "monitor_alarm" in z.files and len(z["monitor_alarm"]):
            a = int(np.asarray(z["monitor_alarm"], bool).sum())
        else:
            a = int((s >= thr).sum())
        n_alarm_forwards += a
        n_alarm_rollouts += int(a > 0)
    return dict(
        run_id=run_id,
        monitor=cfg.get("monitor", "off"),
        capture=cfg.get("capture_activations", "0"),
        n_rollouts=len(merged["rows"]),
        rollouts_per_hour=merged.get("rollouts_per_hour"),
        collisions=sum(bool(r["collision"]) for r in merged["rows"]),
        completions=sum(bool(r["completion"]) for r in merged["rows"]),
        forwards=summ.get("forwards"),
        forward_ms_mean=summ.get("forward_ms_mean"),
        prep_ms_mean=summ.get("prep_ms_mean"),
        capture_ms_mean=summ.get("capture_ms_mean"),
        monitor_ms_mean=summ.get("monitor_ms_mean"),
        threshold=thr,
        monitor_scored=mon.get("scored"),
        monitor_alarm_rate_server=mon.get("alarm_rate"),
        held_rollouts=mon.get("latched_rollouts"),
        alarm_forwards=n_alarm_forwards,
        crossing_forwards=n_cross,
        forwards_with_scores=n_forwards,
        alarm_rate=(n_alarm_forwards / n_forwards) if n_forwards else None,
        crossing_rate=(n_cross / n_forwards) if n_forwards else None,
        debounce=mon.get("debounce"),
        rollouts_with_an_alarm=n_alarm_rollouts,
    )


def main():
    out, ids = sys.argv[1], sys.argv[2:]
    rows = [one(i) for i in ids]
    w = max(len(r["run_id"]) for r in rows)
    print(f"{'run':<{w}}  mode  deb  roll/h   fwd ms  capture  monitor  cross  alarm  held")
    for r in rows:
        print(
            f"{r['run_id']:<{w}}  {r['monitor']:<4}  {r['debounce'] or '-':>3}  "
            f"{(r['rollouts_per_hour'] or 0):6.1f}  "
            f"{(r['forward_ms_mean'] or 0):6.1f}  "
            f"{(r['capture_ms_mean'] or 0):7.4f}  "
            f"{(r['monitor_ms_mean'] or 0):7.4f}  "
            f"{(r['crossing_rate'] or 0):5.3f}  "
            f"{(r['alarm_rate'] or 0):5.3f}  {r['held_rollouts'] or 0:4}"
        )
    with open(out, "w") as fh:
        json.dump(dict(runs=rows), fh, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
