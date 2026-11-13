"""Where exactly does the clearance go to zero? One scene per process, labelled.

The eight October near-plate oa1 scenes that planned but did not certify, plus the one
planner_v2 position that planned but did not certify. For each: the control step, the phase and
the geom pair at the certification failure, and the minimum clearance within every phase.

Why a re-run is needed at all: `step4b_run.run_full` returns before its own `savez` whenever the
executor fails, so none of the eight left a per-step clearance trace. Nothing in it is edited
here -- `dgcore.Episode` is wrapped so the instance survives the early return, and
`dgcore.SafetyMonitor` is wrapped to record the geom pair the harness never writes into its rows.

`Episode.add` already stores, per control step, the clearance in cm (dgcore:1217), the waypoint
label and the actor, so phase attribution needs no new physics -- only the pair does.

SCENE="<shard>:<attempt>" for an October scene, SCENE="v2single" for the planner_v2 position.
Author: Varun Chandra. Env: planner. GPU 1 only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLN = os.path.dirname(HERE)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (PLN, DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
          "/data_new/vchandr2/fastvla-eval",
          "/data_new/vchandr2/vla-safety-interp/src/physics"):
    if p not in sys.path:
        sys.path.insert(0, p)

SCENE = os.environ["SCENE"]
OUTL = os.environ.get("OUTL", os.path.join(HERE, "out_labelled"))
os.makedirs(OUTL, exist_ok=True)
os.environ.setdefault("MODE", "full")
os.environ.setdefault("RES", "256")
os.environ.setdefault("FLOOR_M", "0.02")
if SCENE != "v2single":
    os.environ["TARGETS"] = SCENE
    os.environ["TAGOUT"] = "_lab_" + SCENE.replace(":", "_")

import dgcore as D        # noqa: E402
import step4b_run as S    # noqa: E402
import route as R         # noqa: E402

# ---- phase map over dgcore's own waypoint labels (dgcore:768-774, route's detour_curobo_v*)
def phase_of(label: str) -> str:
    l = str(label or "")
    if l.startswith("pre_grasp") or l.startswith("grasp:"):
        return "grasp approach"
    if l.startswith("close:"):
        return "grasp close"
    if l.startswith("lift:"):
        return "lift"
    if l.startswith("detour_") or l.startswith("transit"):
        return "carry"
    if l.startswith("pre_place:") or l.startswith("place:"):
        return "place"
    if l.startswith("release:") or l.startswith("retreat:"):
        return "release/retreat"
    return l or "unlabelled"


_EPS = []
_Base_ep = D.Episode


class _KeepEpisode(_Base_ep):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _EPS.append(self)


D.Episode = _KeepEpisode
S.D.Episode = _KeepEpisode

_ACC = []
_Base_mon = D.SafetyMonitor


class _PairMonitor(_Base_mon):
    """Records, per control step, the tightest obstacle's geom pair. Degrades to a no-op."""

    def end_step(self, *a, **kw):
        acc = super().end_step(*a, **kw)
        try:
            po = acc.get("per_obstacle") or {}
            if po:
                o, d = min(po.items(), key=lambda kv: kv[1]["min_clearance"])
                _ACC.append(dict(t=acc.get("t"), contact=bool(acc.get("contact")),
                                 min_clearance_cm=100.0 * float(d["min_clearance"]),
                                 obstacle=o, actor=d.get("actor"),
                                 actor_geom=d.get("actor_geom"),
                                 obstacle_geom=d.get("obstacle_geom"),
                                 substep=acc.get("contact_substep")))
        except Exception:
            pass
        return acc


D.SafetyMonitor = _PairMonitor


def summarise(ep, rr):
    n = len(ep.clearance)
    clr = np.asarray(ep.clearance, float)
    lab = list(ep.label)
    ph = [phase_of(x) for x in lab]
    # the certification failure: the first step at which the executed clearance is at its minimum
    jmin = int(np.argmin(clr))
    # pair records align with control steps in order; fall back to clearance matching
    pair = None
    if len(_ACC) == n:
        pair = _ACC[jmin]
    else:
        cand = [a for a in _ACC
                if abs(a["min_clearance_cm"] - float(clr[jmin])) < 1e-6]
        pair = cand[0] if cand else None
    per = {}
    for p, c, l in zip(ph, clr, lab):
        d = per.setdefault(p, dict(phase=p, n=0, min_cm=1e9, label=l))
        d["n"] += 1
        if c < d["min_cm"]:
            d["min_cm"], d["label"] = float(c), l
    order = ["grasp approach", "grasp close", "lift", "carry", "place", "release/retreat"]
    phases = [per[k] for k in order if k in per] + \
             [v for k, v in per.items() if k not in order]
    return dict(
        steps=n,
        fail_step=jmin,
        fail_phase=ph[jmin],
        fail_label=lab[jmin],
        fail_clearance_cm=round(float(clr[jmin]), 4),
        fail_actor_geom=(pair or {}).get("actor_geom"),
        fail_obstacle_geom=(pair or {}).get("obstacle_geom"),
        fail_obstacle=(pair or {}).get("obstacle"),
        pair_aligned=(len(_ACC) == n),
        phases=[dict(phase=p["phase"], steps=p["n"], min_clearance_cm=round(p["min_cm"], 4),
                     at_label=p["label"]) for p in phases],
        first_zero_step=(int(np.argmax(clr <= 0.0)) if bool((clr <= 0.0).any()) else None),
        first_zero_phase=(ph[int(np.argmax(clr <= 0.0))] if bool((clr <= 0.0).any()) else None),
    )


def main():
    R.install("curobo")
    D.CONF["CERTIFY_MIN_M"] = S.FLOOR_M
    if SCENE == "v2single":
        pos = json.load(open(os.path.join(HERE, "kept_positions_single.json")))[0]
        base = {k: np.asarray(v) for k, v in
                dict(np.load(S.BASELINE, allow_pickle=True)).items() if k.endswith("__eef")}
        rr = S.run_full(pos["init"],
                        np.asarray([pos["shift"][0], pos["shift"][1], 0.0], float),
                        base, D.SPEC["names"]["oa1"]["obstacle"], np.random.RandomState(0))
        rr["cid"] = pos["cid"]
    else:
        S.main()
        rr = {}
        rp = os.path.join(S.OUT, f"step4b_full_one{os.environ['TAGOUT']}.json")
        for cand in (rp, os.path.join(S.OUT, f"step4b_full{os.environ['TAGOUT']}.json")):
            if os.path.exists(cand):
                d = json.load(open(cand))
                rr = (d[0] if isinstance(d, list) and d else d) or {}
                break
        rr["cid"] = SCENE
    if not _EPS or not len(_EPS[-1].clearance):
        print(f"[LAB] {SCENE}: no episode captured (planner returned no path?)", flush=True)
        json.dump(dict(scene=SCENE, captured=False, row=rr),
                  open(os.path.join(OUTL, f"lab_{SCENE.replace(':','_')}.json"), "w"),
                  indent=1, default=str)
        print("LABELLED_ONE_DONE", flush=True)
        return
    ep = max(_EPS, key=lambda e: len(e.clearance))
    out = dict(scene=SCENE, captured=True, reason=rr.get("reason"),
               accepted=rr.get("accepted"), executor_ok=rr.get("executor_ok"),
               via_min_cm=rr.get("via_min_cm"), success=rr.get("success"),
               dry_min_clearance_cm=rr.get("dry_min_clearance_cm"),
               **summarise(ep, rr))
    json.dump(out, open(os.path.join(OUTL, f"lab_{SCENE.replace(':','_')}.json"), "w"),
              indent=1, default=str)
    print(f"[LAB] {SCENE}: fail at step {out['fail_step']} in '{out['fail_phase']}' "
          f"({out['fail_label']}) clearance {out['fail_clearance_cm']} cm  "
          f"pair {out['fail_actor_geom']} x {out['fail_obstacle_geom']}", flush=True)
    for p in out["phases"]:
        print(f"      {p['phase']:<16} {p['steps']:>4} steps  min {p['min_clearance_cm']:8.4f} cm",
              flush=True)
    print("LABELLED_ONE_DONE", flush=True)


if __name__ == "__main__":
    main()
