"""M2 step 1: where the 64 ms per clearance call goes, by stage.

`SafetyMonitor._update_clearance` is one function, so the stages inside it are timed by
wrapping its callees rather than by copying it. Two of the splits come for free from the
import structure: `physics_safety` holds its own module-global `gjk_distance` (line 31),
separate from `convex_distance`'s, so the union-hull GJK calls made directly by
`_update_clearance` are distinguishable from the per-pair calls made inside
`DistanceEngine.distance` without any re-entrancy flag. `world_bound` needs one, because
both the actor stage and the obstacle-piece stage call it.

Stages (see DESIGN.md for what each computes):

  S1 actor_bounds     world_bound() per actor geom
  S2 union_hull_xf    _obstacle_hull_world(): transform the obstacle union hull to world
  S3 union_hull_gjk   gjk_distance(actor, union hull), one per actor -- the exact lower
                      bound that lets whole limbs be discarded
  S4 piece_bounds     world_bound() per obstacle piece (248 on oa1)
  S5 exact_pairs      DistanceEngine.distance() on the pairs that survive both prunes
  S5a   of which the GJK core itself
  S5b   of which the world-space hull transform (_w, cached per frame)
  R  residual         numpy lower-bound arithmetic, argsort, Python loop overhead

Env: ACTIONS_NPZ, OUT, [N_CALLS=1000, DISTMAX=0.5, CLEARANCE_EVERY=25]
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP,
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")):
    sys.path.insert(0, p)

import rollout_bootstrap  # noqa: F401, E402

import numpy as np  # noqa: E402

import convex_distance as cd  # noqa: E402
import physics_safety as ps  # noqa: E402
from fastvla.labeler.replay import parse_key, replay_rollout  # noqa: E402

ACTIONS_NPZ = os.environ["ACTIONS_NPZ"]
OUT = os.environ["OUT"]
N_CALLS = int(os.environ.get("N_CALLS", "1000"))
DISTMAX = float(os.environ.get("DISTMAX", "0.5"))
CLEARANCE_EVERY = int(os.environ.get("CLEARANCE_EVERY", "25"))

T = defaultdict(float)
C = Counter()
PER_CALL = []
_in_piece_bounds = {"d": 0}
_cur = {"b": None}


def _b():
    if _cur["b"] is None:
        _cur["b"] = defaultdict(float)
    return _cur["b"]


def _t(name, fn):
    def w(*a, **kw):
        b = _b()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            b[name] += dt
            T[name] += dt
            C[name] += 1
    return w


def _t_flag(name, fn, flag):
    def w(*a, **kw):
        b = _b()
        flag["d"] += 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            flag["d"] -= 1
            b[name] += dt
            T[name] += dt
            C[name] += 1
    return w


def _t_split(base, fn, flag):
    def w(*a, **kw):
        name = f"{base}_piece" if flag["d"] else f"{base}_actor"
        b = _b()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            b[name] += dt
            T[name] += dt
            C[name] += 1
    return w


# S4 sets the flag; world_bound is then attributed to pieces vs actors.
ps.SafetyMonitor._obstacle_bounds = _t_flag(
    "s4_piece_bounds_outer", ps.SafetyMonitor._obstacle_bounds, _in_piece_bounds)
cd.GeomHulls.world_bound = _t_split("world_bound", cd.GeomHulls.world_bound, _in_piece_bounds)
ps.SafetyMonitor._obstacle_hull_world = _t("s2_union_hull_xf",
                                           ps.SafetyMonitor._obstacle_hull_world)
# S3: physics_safety's own module-global reference -> the union-hull calls only.
ps.gjk_distance = _t("s3_union_hull_gjk", ps.gjk_distance)
# S5: convex_distance's reference -> the per-pair core only.
cd.gjk_distance = _t("s5a_gjk_core", cd.gjk_distance)
cd.DistanceEngine._w = _t("s5b_world_xf", cd.DistanceEngine._w)
cd.DistanceEngine.distance = _t("s5_exact_pairs", cd.DistanceEngine.distance)

_uc = ps.SafetyMonitor._update_clearance


def _timed_uc(self, dd, actor_of):
    _cur["b"] = defaultdict(float)
    t0 = time.perf_counter()
    try:
        return _uc(self, dd, actor_of)
    finally:
        tot = time.perf_counter() - t0
        b = _cur["b"]
        b["total"] = tot
        PER_CALL.append(dict(b))
        _cur["b"] = None


ps.SafetyMonitor._update_clearance = _timed_uc

acts_all = dict(np.load(ACTIONS_NPZ, allow_pickle=True))
got, used, pieces = 0, [], {}
for k in sorted(acts_all):
    if got >= N_CALLS:
        break
    a = acts_all[k]
    if a.ndim != 2 or len(a) == 0:
        continue
    tag, ii = parse_key(k)
    before = len(PER_CALL)
    n, mon = replay_rollout(tag, ii, a, distmax=DISTMAX, clearance_every=CLEARANCE_EVERY,
                            on_step=lambda i: len(PER_CALL) - before < N_CALLS - got)
    got = len(PER_CALL)
    pieces[tag] = sum(len(v) for v in mon.obstacle_geoms.values())
    used.append(dict(key=k, control_steps=int(n), calls=len(PER_CALL) - before))
    print(f"[STAGE] {k}: {len(PER_CALL) - before} clearance calls ({got}/{N_CALLS})", flush=True)

PER_CALL = PER_CALL[:N_CALLS]
keys = sorted({k for c in PER_CALL for k in c})
arr = np.array([[c.get(k, 0.0) for k in keys] for c in PER_CALL], dtype=np.float64)
idx = {k: i for i, k in enumerate(keys)}


def col(k):
    return arr[:, idx[k]] if k in idx else np.zeros(len(arr))


total = col("total")
# exclusive stages: s5_exact_pairs already contains s5a and s5b
exclusive = ["world_bound_actor", "s2_union_hull_xf", "s3_union_hull_gjk",
             "world_bound_piece", "s5_exact_pairs"]
resid = total - sum(col(k) for k in exclusive)

STAGES = [
    ("S1  actor world bounds", "world_bound_actor", False),
    ("S2  union hull -> world", "s2_union_hull_xf", False),
    ("S3  union-hull GJK (per actor)", "s3_union_hull_gjk", False),
    ("S4  obstacle piece bounds", "world_bound_piece", False),
    ("S5  exact pairs (DistanceEngine.distance)", "s5_exact_pairs", False),
    ("S5a   of which GJK core", "s5a_gjk_core", True),
    ("S5b   of which world transform (_w)", "s5b_world_xf", True),
]
res = dict(distmax=DISTMAX, clearance_every=CLEARANCE_EVERY, n_calls=len(PER_CALL),
           obstacle_hull_pieces=pieces, rollouts_used=used,
           ms_per_call_total=float(1000 * total.mean()),
           ms_per_call_total_p95=float(1000 * np.percentile(total, 95)),
           calls=dict(C), stages={}, residual_ms=float(1000 * resid.mean()))
print(f"\n[STAGE] {len(PER_CALL)} clearance calls, distmax={DISTMAX}")
print(f"{'stage':44s} {'ms/call':>9s} {'p95':>9s} {'share':>7s} {'calls/call':>11s}")
for label, k, nested in STAGES:
    v = col(k)
    share = "" if nested else f"{100 * v.mean() / total.mean():6.1f}%"
    res["stages"][k] = dict(ms_mean=float(1000 * v.mean()),
                            ms_p95=float(1000 * np.percentile(v, 95)),
                            share_pct=(None if nested else 100 * v.mean() / total.mean()),
                            calls_per_clearance_call=C[k] / max(1, len(PER_CALL)),
                            nested=nested)
    print(f"{label:44s} {1000 * v.mean():9.3f} {1000 * np.percentile(v, 95):9.3f} "
          f"{share:>7s} {C[k] / max(1, len(PER_CALL)):11.1f}")
print(f"{'R   residual (numpy bounds, argsort, glue)':44s} {1000 * resid.mean():9.3f} "
      f"{1000 * np.percentile(resid, 95):9.3f} {100 * resid.mean() / total.mean():6.1f}%")
print(f"{'    TOTAL per clearance call':44s} {1000 * total.mean():9.3f} "
      f"{1000 * np.percentile(total, 95):9.3f}")
print(f"\n[STAGE] x2 for a control step: {2000 * total.mean():.1f} ms/control step")
np.savez_compressed(OUT + ".npz", stages=np.array(keys), per_call=arr)
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"[STAGE] -> {OUT}.json")
print("STAGE BREAKDOWN DONE", flush=True)
