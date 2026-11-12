"""Step 2 of the planner chapter: does the BASE policy collide at the rule-v2 placements?

One closed-loop rollout of checkpoint 17500 at each of the 80 kept candidate positions, with the
fixed monitor (physics_safety.py 5adeedc, gripper fingers now in the watched set) as ground truth.
Nothing is steered and nothing is fine-tuned; this only measures which positions the base policy
actually hits, because only those are worth planning at in Step 3.

The harness is NOT modified. `rollout_openvla_disp.run_one` already takes a `placement` dict whose
shift it applies verbatim, so each candidate is one call with the same seeds, settle and chunking
as every other rollout in this project. `SafetyMonitor` is subclassed only to record WHICH geom
pair caused each scored contact -- the harness never writes that into its rows -- using the
monitor's own substep-resolution record, the same wrapper as run3b.py.

Pre-registered predictions (PREREG_planner_v2.md, amended 0e814f0, before any rollout):
  above 9 cm 90%, 7-9 cm 80%, 5-7 cm 60%, 3-5 cm 35%, overall 65% (52 of 80).

Author: Varun Chandra. Env: learn-gpu, GPU 1.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISP = "/data_new/vchandr2/vla_old_research/vla-safety-interp/src/displacement"
OUT = os.environ.setdefault("OUTDIR", "/data_new/vchandr2/data/planner_v2/step2")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, "/data_new/vchandr2/rollout_stubs")
sys.path.insert(0, "/data_new/vchandr2/scratch/openvla-oft")
sys.path.insert(0, "/data_new/vchandr2/LIBERO-Safety")
sys.path.insert(0, "/data_new/vchandr2/LIBERO-Safety/third_party/robosuite-1.4")
sys.path.insert(0, DISP)
sys.path.insert(0, os.path.join(DISP, "datagen"))

# the unsteered reference trajectories, for the deviation columns; never written back
import shutil  # noqa: E402
for src in ("baseline_17500.npz", "feasible_set.json"):
    dst = os.path.join(OUT, src)
    if not os.path.exists(dst) and os.path.exists(os.path.join(DISP, "out", src)):
        shutil.copyfile(os.path.join(DISP, "out", src), dst)

os.environ.setdefault("STEP", "17500")
os.environ.setdefault("N_INIT", "10")
os.environ.setdefault("OFFSETS", "")
os.environ.setdefault("ARMS", "")
os.environ.setdefault("PROBE", "0")
os.environ.setdefault("WRITE_BASELINE", "0")      # out/baseline_17500.npz stays authoritative
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import rollout_openvla_disp as H  # noqa: E402  (loads the checkpoint at import)

TAG = "oa1"
ROWS = os.path.join(OUT, "rows_step2.json")
PAIRS = os.path.join(OUT, "contact_pairs_step2.json")

_PAIRS, _CUR = {}, {"key": None}
_SM_base = H.SafetyMonitor


class _PairLoggingMonitor(_SM_base):
    """Records the geom pair of every scored contact. Degrades to a no-op on any error."""

    def end_step(self, *a, **kw):
        acc = super().end_step(*a, **kw)
        try:
            if acc and acc.get("contact"):
                k = _CUR["key"] or "unknown"
                for o, d in (acc.get("per_obstacle") or {}).items():
                    if not d.get("contact"):
                        continue
                    rec = dict(step=acc.get("t"), obstacle=o, actor=d.get("actor"),
                               actor_geom=d.get("actor_geom"),
                               obstacle_geom=d.get("obstacle_geom"),
                               substep=acc.get("contact_substep"), max_force=d.get("max_force"))
                    _PAIRS.setdefault(k, [])
                    if not any(r["actor_geom"] == rec["actor_geom"]
                               and r["obstacle_geom"] == rec["obstacle_geom"] for r in _PAIRS[k]):
                        _PAIRS[k].append(rec)
        except Exception:
            pass
        return acc


H.SafetyMonitor = _PairLoggingMonitor


def main():
    t0 = time.time()
    cands = json.load(open(os.path.join(HERE, "candidates_v2.json")))["rows"]
    cands.sort(key=lambda r: (r["init"], -r["intrude_cm"]))
    specs = {int(s["init_idx"]): s for s in H.SPECS if s["tag"] == TAG}
    assert len(specs) == 10, sorted(specs)
    print(f"[STEP2] {len(cands)} candidates over {len(specs)} inits, "
          f"checkpoint {H.STEP}, fixed monitor", flush=True)

    rows, done, seen = [], 0, {}
    for c in cands:
        ii = int(c["init"])
        seen[ii] = seen.get(ii, -1) + 1
        cid = f"{TAG}__i{ii:02d}__c{seen[ii]}"      # c0 = deepest intrusion for that init
        _CUR["key"] = cid
        pl = dict(feasible=True, clamped=False, skipped=False, mode="rule_v2",
                  shift=[float(c["shift"][0]), float(c["shift"][1]), 0.0],
                  offset_m=float(np.hypot(c["shift"][0], c["shift"][1])),
                  achieved_m=float(np.hypot(c["shift"][0], c["shift"][1])),
                  intrusion_m=float(c["intrude_cm"]) / 100.0)
        r, skip = H.run_one(specs[ii], 0.0, f"c{done:03d}", False, placement=pl)
        done += 1
        if r is None:
            print(f"[STEP2] SKIP {cid} reason={(skip or {}).get('reason')}", flush=True)
            rows.append(dict(cid=cid, init=ii, skipped=True,
                             reason=(skip or {}).get("reason")))
            continue
        res = r["res"]
        row = dict(cid=cid, init=ii, x=c["x"], y=c["y"], bin=c["bin"],
                   intrude_cm=c["intrude_cm"], ranked_by=c["ranked_by"],
                   d_ee_cm=c["d_ee_cm"], d_l5_cm=c["d_l5_cm"],
                   collision=bool(res["collision"]), completion=bool(res["completion"]),
                   min_clearance_cm=res["min_clearance_cm"], steps=res["steps"],
                   collision_step=res.get("collision_step"), obstacle=res.get("obstacle"),
                   pairs=_PAIRS.get(cid, []), skipped=False)
        rows.append(row)
        print(f"[STEP2] {done:3d}/{len(cands)} i{ii:02d} {c['bin']:<8} "
              f"intr {c['intrude_cm']:5.2f}  collision {str(row['collision']):5} "
              f"complete {str(row['completion']):5}  minclr {row['min_clearance_cm']:7.3f} cm"
              + (f"  pair {row['pairs'][0]['actor_geom']} x {row['pairs'][0]['obstacle_geom']}"
                 if row["pairs"] else ""), flush=True)
        if done % 10 == 0:
            json.dump(rows, open(ROWS, "w"), indent=1)
            json.dump(_PAIRS, open(PAIRS, "w"), indent=1)
    json.dump(rows, open(ROWS, "w"), indent=1)
    json.dump(_PAIRS, open(PAIRS, "w"), indent=1)
    print(f"[STEP2] done in {time.time()-t0:.0f}s", flush=True)
    print("STEP2_DONE", flush=True)


if __name__ == "__main__":
    main()
