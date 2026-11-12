"""Step 3 of the planner chapter: can cuRobo certify where the base policy collides?

ONE position per process. The stale-attachment bug (RESULTS_4B.md) means a long-lived MotionGen
carries the previous scene's attached payload into the next plan, so every position gets a fresh
process and nothing can carry; `run_step3.sh` is the sequential driver.

Nothing here re-implements the planner or the certifier. `step4b_run.run_full` already does the
whole job -- install the cuRobo router over dgcore.plan_detours, build the scene at a given shift,
plan the carried traverse, execute with the OSC executor, run the 2 cm GJK gate with the FIXED
monitor (physics_safety.py 5adeedc) and the rendered-replay gate -- and already records the RAW
minimum clearance of every plan in `dry_min_clearance_cm`, not just the certified flag. This file
chooses the positions, calls it once, and writes the row plus a LeRobot episode when certified.

Floors, as instructed, unchanged:
  D.CONF["CERTIFY_MIN_M"] = 0.02   the real 2 cm certification (FLOOR_M below)
  route.CERTIFY_FLOOR_M   = 0.01   the planner-side terminal-gap pre-filter, left at its default

Author: Varun Chandra. Env: planner. GPU 1 only.
Env: POS_IDX (row index into kept_positions.json), KEPT (path), OUT3, FLOOR_M=0.02
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLN = os.path.dirname(HERE)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (PLN, DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
           "/data_new/vchandr2/fastvla-eval",
           "/data_new/vchandr2/vla-safety-interp/src/physics"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT3 = os.environ.get("OUT3", os.path.join(HERE, "out_step3"))
KEPT = os.environ.get("KEPT", os.path.join(HERE, "kept_positions.json"))
os.makedirs(OUT3, exist_ok=True)
os.environ.setdefault("FLOOR_M", "0.02")       # the real certification, not the planner pre-filter
os.environ.setdefault("MODE", "full")
os.environ.setdefault("RES", "256")

import dgcore as D       # noqa: E402
import step4b_run as S   # noqa: E402
import route as R        # noqa: E402

TAG = "oa1"


def main():
    idx = int(os.environ["POS_IDX"])
    kept = json.load(open(KEPT))
    if idx >= len(kept):
        print(f"[STEP3] index {idx} past the end ({len(kept)} positions)", flush=True)
        return
    pos = kept[idx]
    t0 = time.perf_counter()
    R.install("curobo")
    D.CONF["CERTIFY_MIN_M"] = S.FLOOR_M
    base = {k: np.asarray(v) for k, v in
            dict(np.load(S.BASELINE, allow_pickle=True)).items() if k.endswith("__eef")}
    obstacle = D.SPEC["names"][TAG]["obstacle"]
    shift = np.asarray([pos["shift"][0], pos["shift"][1], 0.0], float)
    print(f"[STEP3] {idx:02d} {pos['cid']}  init {pos['init']}  bin {pos['bin']}  "
          f"intrusion {pos['intrude_cm']:.2f} cm  shift {100*np.linalg.norm(shift):.2f} cm  "
          f"certify floor {100*S.FLOOR_M:.1f} cm  planner floor "
          f"{100*getattr(R, 'CERTIFY_FLOOR_M', float('nan')):.1f} cm", flush=True)

    rr = S.run_full(pos["init"], shift, base, obstacle, np.random.RandomState(0))
    rr.update(cid=pos["cid"], init=pos["init"], bin=pos["bin"],
              intrude_cm=pos["intrude_cm"], ranked_by=pos["ranked_by"],
              x=pos["x"], y=pos["y"], base_policy_min_clearance_cm=pos["min_clearance_cm"],
              base_policy_pair=(pos.get("pairs") or [{}])[0].get("actor_geom"),
              seconds=round(time.perf_counter() - t0, 1))
    # outcome in the three categories the chapter asked for, with the reason kept verbatim
    if not rr.get("planned"):
        rr["outcome"] = "no_plan"
    elif rr.get("accepted"):
        rr["outcome"] = "certified"
    else:
        rr["outcome"] = "planned_not_certified"
    json.dump(rr, open(os.path.join(OUT3, f"row_{idx:03d}.json"), "w"), indent=1, default=str)
    print(f"[STEP3] {idx:02d} {rr['outcome']}  raw min clearance "
          f"{rr.get('dry_min_clearance_cm')} cm  reason {rr.get('reason')}  "
          f"{rr['seconds']:.0f}s", flush=True)
    print("STEP3_ONE_DONE", flush=True)


if __name__ == "__main__":
    main()
