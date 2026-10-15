"""Manifests: which (task, init, arm) rollouts a run covers, and the scene for each.

Two sets matter for M3:

  "ctrl60"  the 60 fixed starts, no displacement -- the reference 29/60 collisions and
            25 or 26 completions. Scene setup is the driver's unshifted branch.
  "gate10"  oa1 inits 0-9 at the intrusion-5 placement from out/feasible_set.json, which
            is the 12-minute gate's set: 10/10 collisions, 0.00 cm clearance, completions
            on starts 4, 6, 9. Scene setup is the driver's `placement` branch, in the
            order datagen/curve_gate.py applies it.
  "intr5"   the full intrusion-5 set: every (task, init) whose feasible_set.json arm "5"
            is feasible -- oa1 10, oa2 10, oa3 10, oa4 8, oa5 10, oa6_L1 0 = 48 starts.
            gate10 is its oa1 subset. The reference for checkpoint 17500 is
            out_instruct/rows_I0_intr5.json.

The placement is applied VERBATIM from feasible_set.json and nothing is re-solved, which
is what the driver does on this path ("the survey already decided feasibility and fixed
the geometry", rollout_openvla_disp.py run_one). geom_shift.solve_intrusion is what
produced those shifts originally; re-solving here would risk a different answer from a
different baseline trajectory.

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


def _intr5_rows():
    with open(FEASIBLE) as fh:
        rows = json.load(fh)["rows"]
    out = []
    for r in rows:
        arm = (r.get("arms") or {}).get("5") or {}
        if not (arm.get("shift") and arm.get("feasible")):
            continue
        t = BY_TAG.get(r["tag"])
        if t is None:
            continue
        i = int(r["init_idx"])
        out.append(
            dict(
                tag=r["tag"],
                level=t["level"],
                lid=t["lid"],
                init_idx=i,
                arm="intr5",
                shift=list(arm["shift"]),
                key=rollout_key(r["tag"], i, "intr5"),
            )
        )
    return out


def intr5():
    """The 48 feasible intrusion-5 starts."""
    return sorted(_intr5_rows(), key=lambda r: (r["tag"], r["init_idx"]))


def unshifted(inits):
    """Unshifted starts over a given range of inits, for activation collection."""
    out = []
    for t in TASKS:
        for i in inits:
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


MANIFESTS = {
    "ctrl60": ctrl60,
    "gate10": gate10,
    "intr5": intr5,
    # activation-collection sets (M7b step 1). Train and test are disjoint by init:
    # training uses inits 10-49, test uses 0-9, so no scene appears in both.
    "train_unshifted": lambda: unshifted(range(10, 50)),
    "test_unshifted": lambda: unshifted(range(0, 10)),
}


def build(name):
    if name not in MANIFESTS:
        raise ValueError(f"unknown manifest {name!r}; have {sorted(MANIFESTS)}")
    return MANIFESTS[name]()
