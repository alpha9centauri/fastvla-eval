"""Manifests: which (task, init, arm) rollouts a run covers, and the scene for each.

Two sets matter for M3:

  "ctrl60"  the 60 fixed starts, no displacement -- the reference 29/60 collisions and
            25 or 26 completions. Scene setup is the driver's unshifted branch.
  "gate10"  oa1 inits 0-9 at the intrusion-5 placement from out/feasible_set.json, which
            is the 12-minute gate's set: 10/10 collisions, 0.00 cm clearance, completions
            on starts 4, 6, 9. Scene setup is the driver's `placement` branch, in the
            order datagen/curve_gate.py applies it.

Placements are read from the research tree read-only. Nothing here writes outside
fastvla-eval.
"""

import json
import os

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
FEASIBLE = os.path.join(DISP, "out", "feasible_set.json")

TASKS = [
    dict(suite="obstacle_avoidance", level=0, lid=0, tag="oa1"),
    dict(suite="obstacle_avoidance", level=0, lid=1, tag="oa2"),
    dict(suite="obstacle_avoidance", level=0, lid=2, tag="oa3"),
    dict(suite="obstacle_avoidance", level=0, lid=3, tag="oa4"),
    dict(suite="obstacle_avoidance", level=0, lid=4, tag="oa5"),
    dict(suite="obstacle_avoidance", level=1, lid=0, tag="oa6_L1"),
]
BY_TAG = {t["tag"]: t for t in TASKS}


def rollout_key(tag, init_idx, arm):
    return f"{tag}__i{int(init_idx):02d}__{arm}"


def ctrl60():
    out = []
    for t in TASKS:
        for i in range(10):
            out.append(
                dict(
                    tag=t["tag"],
                    level=t["level"],
                    lid=t["lid"],
                    init_idx=i,
                    arm="base",
                    shift=None,
                    key=rollout_key(t["tag"], i, "base"),
                )
            )
    return out


def gate10():
    with open(FEASIBLE) as fh:
        rows = json.load(fh)["rows"]
    fs = {r["init_idx"]: ((r.get("arms") or {}).get("5") or {}) for r in rows if r["tag"] == "oa1"}
    out = []
    t = BY_TAG["oa1"]
    for i in range(10):
        arm = fs.get(i, {})
        if not arm.get("shift"):
            continue
        out.append(
            dict(
                tag="oa1",
                level=t["level"],
                lid=t["lid"],
                init_idx=i,
                arm="intr5",
                shift=list(arm["shift"]),
                key=rollout_key("oa1", i, "intr5"),
            )
        )
    return out


MANIFESTS = {"ctrl60": ctrl60, "gate10": gate10}


def build(name):
    if name not in MANIFESTS:
        raise ValueError(f"unknown manifest {name!r}; have {sorted(MANIFESTS)}")
    return MANIFESTS[name]()
