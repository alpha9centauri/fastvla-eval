"""Step 3 end to end on the three step-2 scenes: does a plan come out, and does it certify?

Runs dgcore.generate_demo with PLANNER=curobo on the same eval-rule oa1 scenes step 2 measured
(inits 0, 1, 2), through the unmodified downstream path: OSC executor, 2 cm GJK certification,
rendered-replay gate, writer. Reports what was asked for -- whether plan_single returns a path,
the time per plan, and the GJK clearance of the executed demo.

Author: Varun Chandra
Env: planner. GPU 1 only.
Env vars: INITS=0,1,2  PLANNER=curobo
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (
    DISP,
    os.path.join(DISP, "datagen"),
    "/data_new/vchandr2",
    "/data_new/vchandr2/fastvla-eval",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D  # noqa: E402

from fastvla.planner import route as R  # noqa: E402

TAG = "oa1"
INITS = [int(x) for x in os.environ.get("INITS", "0,1,2").split(",")]
PLANNER = os.environ.get("PLANNER", "curobo")
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")


def main():
    installed = R.install(PLANNER)
    print(f"planner installed: {installed}")
    base = {
        k: np.asarray(v)
        for k, v in dict(np.load(BASELINE, allow_pickle=True)).items()
        if k.endswith("__eef")
    }

    rows = []
    for ii in INITS:
        R.reset_stats()
        rng = np.random.RandomState(ii)
        t0 = time.perf_counter()
        r = D.generate_demo(TAG, ii, "varied", rng, res=84, placement="eval_rule", baseline=base)
        wall = time.perf_counter() - t0
        st = R.stats()
        plan_s = sum(x.get("seconds", 0.0) for x in st if x.get("stage") == "plan")
        planned = any(x.get("ok") for x in st if x.get("stage") == "plan")
        det = (r.get("detours") or [{}])[0] if r.get("detours") else {}
        row = dict(
            init=ii,
            accepted=bool(r["accepted"]),
            reason=r["reason"],
            shift_cm=(
                round(100 * float(np.linalg.norm(r["shift"])), 2) if r.get("shift") else None
            ),
            plan_returned_path=planned,
            plan_seconds=round(plan_s, 4),
            demo_seconds=round(wall, 1),
            n_via=det.get("n_via"),
            handoff=det.get("handoff"),
            payload_attached=det.get("payload_attached"),
            tilt_deg=det.get("tilt_deg"),
            dry_min_clearance_cm=r.get("min_clearance_cm"),
            replay_min_clearance_cm=r.get("replay_min_clearance_cm"),
            replay_success=r.get("replay_success"),
            steps=(len(r["episode"]) if r.get("episode") is not None else None),
            stats=st,
        )
        rows.append(row)
        print(
            f"  [i{ii:02d}] shift {row['shift_cm']}cm  plan={planned} "
            f"{1000 * plan_s:.0f} ms  accepted={row['accepted']}  "
            f"reason={row['reason'] or '-'}  dry clr={row['dry_min_clearance_cm']}  "
            f"replay clr={row['replay_min_clearance_cm']}",
            flush=True,
        )

    print()
    print("=" * 100)
    print(f"STEP 3 END TO END -- {TAG}, PLANNER={installed}, {len(rows)} scenes")
    print("=" * 100)
    print(
        f"  {'init':>5} {'shift cm':>9} {'path':>6} {'plan ms':>8} {'via':>4} "
        f"{'handoff up/out cm':>18} {'tilt':>6} {'dry clr cm':>11} {'replay clr cm':>14} "
        f"{'accepted':>9}"
    )
    for r in rows:
        h = r["handoff"] or {}
        hs = f"{100 * h.get('up_m', 0):.0f}/{100 * h.get('out_m', 0):.0f}" if h else "-"
        print(
            f"  {r['init']:>5} {str(r['shift_cm']):>9} {str(r['plan_returned_path']):>6} "
            f"{1000 * r['plan_seconds']:>8.0f} {str(r['n_via'] or '-'):>4} {hs:>18} "
            f"{str(r['tilt_deg'] or '-'):>6} {str(r['dry_min_clearance_cm']):>11} "
            f"{str(r['replay_min_clearance_cm']):>14} {str(r['accepted']):>9}"
        )
    n_path = sum(r["plan_returned_path"] for r in rows)
    n_acc = sum(r["accepted"] for r in rows)
    ps = [r["plan_seconds"] for r in rows if r["plan_returned_path"]]
    print()
    print(f"  plan_single returned a path   {n_path}/{len(rows)}")
    if ps:
        print(
            f"  time per plan                 mean {1000 * np.mean(ps):.0f} ms, "
            f"range {1000 * min(ps):.0f}-{1000 * max(ps):.0f} ms"
        )
    print(f"  certified demos (accepted)    {n_acc}/{len(rows)}")
    clr = [r["replay_min_clearance_cm"] for r in rows if r["replay_min_clearance_cm"] is not None]
    if clr:
        print(
            f"  replay GJK clearance          {', '.join(f'{c:.2f}' for c in clr)} cm (floor 2.00)"
        )
    for r in rows:
        if not r["accepted"]:
            print(f"  init {r['init']} rejected: {r['reason']}")
    with open(os.path.join(OUT, "verify_step3.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=str)
    print("VERIFY_STEP3_DONE")


if __name__ == "__main__":
    main()
