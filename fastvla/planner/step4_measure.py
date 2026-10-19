"""Step 4: the planner on all 197 eval-rule oa1 scenes of the 2b smoke.

Reproduces the 2b smoke's scene sequence exactly rather than approximately. That smoke ran
`gen_stagec.py` with TAG=oa1, VARIANTS=varied, N_PER=10, SHARDS=10, MAX_MULT=20, SEED=0,
PLACEMENT=eval_rule and the 50-init baseline, giving each shard k an rng seeded
`SEED + 7919*0 + 131*k` and at most 20 attempts. It accepted 1 of 197.

This driver mirrors that loop serially in one process -- one warm cuRobo instance instead of ten
spawned ones -- and does NOT stop a shard early on acceptance, so every shard runs its full 20
draws. 200 scenes result: 197 are the identical draws the smoke made, and 3 are the continuation
of shard 4, the only shard that accepted and therefore stopped short. Both numbers are reported.

Acceptance at two floors from one run: dgcore's certification floor is set to 1 cm for the run and
the executed demo's own minimum GJK clearance is recorded, so acceptance at 2 cm is recovered
exactly by counting demos whose clearance also clears 2 cm. The 2 cm certification is not weakened
-- it is applied afterwards to the same measurements.

Author: Varun Chandra
Env: planner. GPU 1 only.
Env vars: SHARDS=10 ATTEMPTS=20 SEED=0 FLOOR_M=0.01 PLANNER=curobo
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CLIPS = os.path.join(HERE, "clips")
for d in (OUT, CLIPS):
    os.makedirs(d, exist_ok=True)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
           "/data_new/vchandr2/fastvla-eval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D                                  # noqa: E402
from fastvla.planner import route as R               # noqa: E402

TAG = "oa1"
SHARDS = int(os.environ.get("SHARDS", "10"))
ATTEMPTS = int(os.environ.get("ATTEMPTS", "20"))
SEED = int(os.environ.get("SEED", "0"))
FLOOR_M = float(os.environ.get("FLOOR_M", "0.01"))
PLANNER = os.environ.get("PLANNER", "curobo")
RES = int(os.environ.get("RES", "256"))
BASELINE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")
# shard 4 accepted at attempt 17 in the 2b smoke and stopped; its draws 18-20 are new here
SMOKE_STOPPED = {4: 17}


def cause(row):
    """One failure bucket per scene, from the planner's stats and dgcore's own reason."""
    if row["accepted"]:
        return "accepted"
    st = row["stats"]
    reason = row["reason"] or ""
    if any(s.get("stage") == "tilt_reject" for s in st):
        return "tilt"
    plan = next((s for s in st if s.get("stage") == "plan"), None)
    if plan is not None and not plan.get("ok"):
        return "no_clear_handoff" if plan.get("n_plan_calls", 0) == 0 else "plan_fail"
    if any(s.get("stage") == "terminal" for s in st):
        return "scene_terminal"
    if reason.startswith("scene_reject"):
        return "scene_reject"
    if reason.startswith("gjk_below") or "clearance" in reason:
        return "certification"
    if "unreached" in reason or "success_predicate" in reason:
        return "executor"
    if reason.startswith("planner_failure:exception"):
        return "exception"
    return "other"


def main():
    installed = R.install(PLANNER)
    prev_floor = D.CONF["CERTIFY_MIN_M"]
    D.CONF["CERTIFY_MIN_M"] = FLOOR_M
    print(f"planner={installed}  certification floor for the run {100 * FLOOR_M:.1f} cm "
          f"(was {100 * prev_floor:.1f}); acceptance at 2 cm recovered from the recorded "
          f"clearances", flush=True)
    base = {k: np.asarray(v) for k, v in
            dict(np.load(BASELINE, allow_pickle=True)).items() if k.endswith("__eef")}
    print(f"baseline trajectories: {len(base)}", flush=True)

    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    rows, t_start = [], time.perf_counter()
    for k in range(SHARDS):
        rng = np.random.RandomState(SEED + 7919 * 0 + 131 * k)
        for a in range(ATTEMPTS):
            init_idx = int(rng.randint(lo, hi + 1))
            R.reset_stats()
            t0 = time.perf_counter()
            r = D.generate_demo(TAG, init_idx, "varied", rng, res=RES,
                                placement="eval_rule", baseline=base)
            wall = time.perf_counter() - t0
            st = R.stats()
            plan = next((s for s in st if s.get("stage") == "plan"), {}) or {}
            det = (r.get("detours") or [{}])[0] if r.get("detours") else {}
            tilt = det.get("tilt_deg")
            if tilt is None:
                tr = next((s for s in st if s.get("stage") == "tilt_reject"), None)
                tilt = tr.get("tilt_deg") if tr else None
            h = det.get("handoff") or (plan.get("handoff") or {})
            row = dict(
                shard=k, attempt=a, in_smoke=bool(a < SMOKE_STOPPED.get(k, ATTEMPTS)),
                init_idx=init_idx, accepted=bool(r["accepted"]), reason=r["reason"],
                shift_cm=(round(100 * float(np.linalg.norm(r["shift"])), 3)
                          if r.get("shift") else None),
                intrusion_cm=(round(100 * r["intrusion_m"], 2)
                              if r.get("intrusion_m") is not None else None),
                plan_ok=bool(plan.get("ok", False)),
                plan_seconds_total=plan.get("seconds"),
                plan_seconds_single=plan.get("seconds_single"),
                n_ik_tried=plan.get("n_ik_tried"), n_ik_accepted=plan.get("n_ik_accepted"),
                n_plan_calls=plan.get("n_plan_calls"),
                handoff_up_cm=(round(100 * h["up_m"], 1) if h.get("up_m") is not None else None),
                handoff_out_cm=(round(100 * h["out_m"], 1) if h.get("out_m") is not None else None),
                tilt_deg=tilt, payload_attached=det.get("payload_attached"),
                n_via=det.get("n_via"),
                dry_min_clearance_cm=r.get("min_clearance_cm"),
                replay_min_clearance_cm=r.get("replay_min_clearance_cm"),
                replay_success=r.get("replay_success"),
                steps=(len(r["episode"]) if r.get("episode") is not None else None),
                scene_hash=r.get("scene_hash"), demo_seconds=round(wall, 2), stats=st)
            row["cause"] = cause(row)
            rows.append(row)
            if (len(rows) % 10 == 0) or row["accepted"]:
                acc = sum(x["accepted"] for x in rows)
                print(f"[{len(rows):>3}/{SHARDS * ATTEMPTS}] s{k} a{a} init {init_idx:>2} "
                      f"shift {row['shift_cm']}cm  {row['cause']:>16}  "
                      f"clr {row['replay_min_clearance_cm']}  accepted so far {acc}  "
                      f"({time.perf_counter() - t_start:.0f}s)", flush=True)
            json.dump([{kk: vv for kk, vv in x.items() if kk != "stats"} for x in rows],
                      open(os.path.join(OUT, "step4_rows.json"), "w"), indent=1, default=str)
    D.CONF["CERTIFY_MIN_M"] = prev_floor
    json.dump(rows, open(os.path.join(OUT, "step4_rows_full.json"), "w"), indent=1, default=str)
    report(rows, time.perf_counter() - t_start)


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / max(1, d):.1f}%"


def report(rows, wall):
    smoke = [r for r in rows if r["in_smoke"]]
    acc1 = [r for r in rows if r["accepted"]]
    acc2 = [r for r in acc1 if (r["replay_min_clearance_cm"] or 0) >= 2.0]
    s_acc1 = [r for r in smoke if r["accepted"]]
    s_acc2 = [r for r in s_acc1 if (r["replay_min_clearance_cm"] or 0) >= 2.0]

    L = []
    A = L.append
    A("=" * 100)
    A(f"STEP 4 -- cuRobo planner on the 2b smoke's eval-rule oa1 scenes")
    A("=" * 100)
    A(f"  scenes run              {len(rows)}  ({len(smoke)} identical to the 2b smoke's draws, "
      f"{len(rows) - len(smoke)} the continuation of the shard that stopped early)")
    A(f"  wall clock              {wall / 60:.1f} min  "
      f"({wall / max(1, len(rows)):.1f} s per scene)")
    A("")
    A("  ACCEPTANCE")
    A(f"    at 1 cm certification   {pct(len(acc1), len(rows))}      "
      f"(2b smoke subset: {pct(len(s_acc1), len(smoke))})")
    A(f"    at 2 cm certification   {pct(len(acc2), len(rows))}      "
      f"(2b smoke subset: {pct(len(s_acc2), len(smoke))})")
    A(f"    scripted planner baseline, same scenes, 2 cm: 1/197 = 0.5%")
    A("")
    A("  PLANNING TIME (scenes where a path was returned)")
    ok = [r for r in rows if r["plan_ok"]]
    for nm, key in (("single successful call", "plan_seconds_single"),
                    ("total across candidates", "plan_seconds_total")):
        v = np.asarray([r[key] for r in ok if r.get(key) is not None], float)
        if len(v):
            A(f"    {nm:<24} mean {v.mean():.2f} s   median {np.median(v):.2f} s   "
              f"range {v.min():.2f}-{v.max():.2f} s   n={len(v)}")
    it = np.asarray([r["n_ik_tried"] for r in rows if r.get("n_ik_tried")], float)
    ia = np.asarray([r["n_ik_accepted"] for r in rows if r.get("n_ik_accepted") is not None], float)
    if len(it):
        A(f"    hand-off candidates tried per scene: mean {it.mean():.1f} of 24; "
          f"accepted by collision-aware IK: mean {ia.mean():.2f}")
    dm = np.asarray([r["demo_seconds"] for r in rows], float)
    A(f"    time per demo (whole scene)  mean {dm.mean():.1f} s   median "
      f"{np.median(dm):.1f} s   range {dm.min():.1f}-{dm.max():.1f} s")
    A("")
    A("  ACCEPTED DEMOS -- hand-off and tilt")
    if acc1:
        A(f"    {'shard':>5} {'att':>4} {'init':>5} {'shift cm':>9} {'up cm':>6} {'out cm':>7} "
          f"{'tilt deg':>9} {'via':>4} {'dry clr':>8} {'replay clr':>11} {'steps':>6}")
        for r in sorted(acc1, key=lambda x: -(x["replay_min_clearance_cm"] or 0)):
            A(f"    {r['shard']:>5} {r['attempt']:>4} {r['init_idx']:>5} "
              f"{str(r['shift_cm']):>9} {str(r['handoff_up_cm']):>6} "
              f"{str(r['handoff_out_cm']):>7} {str(r['tilt_deg']):>9} {str(r['n_via']):>4} "
              f"{str(r['dry_min_clearance_cm']):>8} "
              f"{str(r['replay_min_clearance_cm']):>11} {str(r['steps']):>6}")
        for nm, key in (("hand-off height cm", "handoff_up_cm"),
                        ("hand-off offset cm", "handoff_out_cm"),
                        ("wrist tilt deg", "tilt_deg"),
                        ("replay GJK clearance cm", "replay_min_clearance_cm")):
            v = np.asarray([r[key] for r in acc1 if r.get(key) is not None], float)
            if len(v):
                A(f"    {nm:<26} mean {v.mean():7.2f}  median {np.median(v):7.2f}  "
                  f"range {v.min():.2f}-{v.max():.2f}")
    else:
        A("    none")
    A("")
    A("  FAILURE BREAKDOWN BY CAUSE")
    c = collections.Counter(r["cause"] for r in rows)
    names = dict(no_clear_handoff="no clear hand-off (collision-aware IK refused all 24)",
                 plan_fail="plan fail (IK accepted a hand-off, plan_single found no path)",
                 tilt="tilt above the limit", executor="executor (waypoint unreached / predicate)",
                 certification="certification (GJK below the floor)",
                 scene_terminal="scene: terminal pose inside the floor",
                 scene_reject="scene rejected before planning", exception="exception",
                 accepted="accepted", other="other")
    for k2, n in c.most_common():
        A(f"    {names.get(k2, k2):<58} {n:>4}  {100.0 * n / len(rows):>5.1f}%")
    A("")
    tilts = np.asarray([r["tilt_deg"] for r in rows if r.get("tilt_deg") is not None], float)
    if len(tilts):
        A(f"  tilt over every scene with a path: mean {tilts.mean():.1f} deg, "
          f"median {np.median(tilts):.1f}, range {tilts.min():.1f}-{tilts.max():.1f}, "
          f"n={len(tilts)} (limit {R.TILT_LIMIT_DEG:.0f})")
    txt = "\n".join(L)
    print()
    print(txt)
    open(os.path.join(OUT, "step4_report.txt"), "w").write(txt + "\n")
    print("STEP4_DONE")


if __name__ == "__main__":
    main()
