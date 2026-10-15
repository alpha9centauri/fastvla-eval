"""M2 step 3: C++ vs Python parity on real saved states, per GJK call.

Replays the same rollouts as audit Table 4 and, for every GJK call the labeler makes,
evaluates BOTH implementations on the same inputs and records the difference. Both call
sites are covered, because `physics_safety` and `convex_distance` hold separate
module-global references to `gjk_distance`:

  * S3, the union-hull lower bound (`physics_safety.gjk_distance`)
  * S5a, the exact per-pair core (`convex_distance.gjk_distance`)

Acceptance is max |difference| < 1e-6 m (plan section 6, "Labeler parity"). Also recorded:
whether the two ever disagree about intersection (one returns 0.0 and the other does not),
which matters more than the magnitude -- 0.0 is the signal that a pair is touching.

Env: ACTIONS_NPZ, OUT, [N_CALLS=1000, DISTMAX=0.5, CLEARANCE_EVERY=25]
"""

import json
import os
import sys

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
):
    sys.path.insert(0, p)

import convex_distance as cd  # noqa: E402
import numpy as np  # noqa: E402
import physics_safety as ps  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402

from fastvla.labeler import load_fastgjk  # noqa: E402
from fastvla.labeler.replay import parse_key, replay_rollout  # noqa: E402

fastgjk = load_fastgjk()

ACTIONS_NPZ = os.environ["ACTIONS_NPZ"]
OUT = os.environ["OUT"]
N_CALLS = int(os.environ.get("N_CALLS", "1000"))
DISTMAX = float(os.environ.get("DISTMAX", "0.5"))
CLEARANCE_EVERY = int(os.environ.get("CLEARANCE_EVERY", "25"))

REC = {"s3": [], "s5": []}  # (python, cpp) per call
_py_gjk = cd.gjk_distance  # the single reference implementation


def _both(site):
    def w(A, B, max_iter=64, tol=1e-12):
        p = _py_gjk(A, B, max_iter, tol)
        c = fastgjk.gjk_distance(A, B, max_iter, tol)
        REC[site].append((p, c))
        return p  # the Python value is what the run uses, so the
        # replay itself is unchanged by measuring it

    return w


ps.gjk_distance = _both("s3")
cd.gjk_distance = _both("s5")

acts_all = dict(np.load(ACTIONS_NPZ, allow_pickle=True))
n_clear = {"n": 0}
_uc = ps.SafetyMonitor._update_clearance


def _count(self, dd, actor_of):
    n_clear["n"] += 1
    return _uc(self, dd, actor_of)


ps.SafetyMonitor._update_clearance = _count

used = []
for k in sorted(acts_all):
    if n_clear["n"] >= N_CALLS:
        break
    a = acts_all[k]
    if a.ndim != 2 or len(a) == 0:
        continue
    tag, ii = parse_key(k)
    before = n_clear["n"]
    n, _mon = replay_rollout(
        tag,
        ii,
        a,
        distmax=DISTMAX,
        clearance_every=CLEARANCE_EVERY,
        on_step=lambda i: n_clear["n"] < N_CALLS,
    )
    used.append(dict(key=k, control_steps=int(n), clearance_calls=n_clear["n"] - before))
    print(
        f"[PARITY] {k}: {n_clear['n'] - before} clearance calls ({n_clear['n']}/{N_CALLS})",
        flush=True,
    )

res = dict(
    distmax=DISTMAX,
    clearance_every=CLEARANCE_EVERY,
    n_clearance_calls=n_clear["n"],
    rollouts_used=used,
    sites={},
)
worst_overall = 0.0
for site, label in (("s3", "S3 union-hull GJK"), ("s5", "S5a exact-pair GJK core")):
    A = np.array(REC[site], dtype=np.float64)
    if not len(A):
        continue
    p, c = A[:, 0], A[:, 1]
    d = np.abs(p - c)
    # disagreement about intersection: one says exactly 0.0, the other does not
    zp, zc = (p == 0.0), (c == 0.0)
    rel = np.zeros_like(d)
    nz = p != 0.0
    rel[nz] = d[nz] / np.abs(p[nz])
    res["sites"][site] = dict(
        label=label,
        n_calls=int(len(A)),
        max_abs_diff_m=float(d.max()),
        mean_abs_diff_m=float(d.mean()),
        p95_abs_diff_m=float(np.percentile(d, 95)),
        max_rel_diff=float(rel.max()),
        n_exact_equal=int((d == 0.0).sum()),
        n_zero_python=int(zp.sum()),
        n_zero_cpp=int(zc.sum()),
        n_intersection_disagreements=int((zp != zc).sum()),
        python_min=float(p.min()),
        python_max=float(p.max()),
    )
    worst_overall = max(worst_overall, float(d.max()))
    np.savez_compressed(f"{OUT}_{site}.npz", pairs=A)
    r = res["sites"][site]
    print(f"\n[PARITY] {label}: {r['n_calls']} calls")
    print(f"  max |py - c++|          {r['max_abs_diff_m']:.3e} m")
    print(f"  mean |py - c++|         {r['mean_abs_diff_m']:.3e} m")
    print(f"  p95 |py - c++|          {r['p95_abs_diff_m']:.3e} m")
    print(f"  max relative diff       {r['max_rel_diff']:.3e}")
    print(f"  bit-identical calls     {r['n_exact_equal']}/{r['n_calls']}")
    print(f"  zero (touching) py/c++  {r['n_zero_python']}/{r['n_zero_cpp']}")
    print(f"  intersection disagreements  {r['n_intersection_disagreements']}")

res["max_abs_diff_m_overall"] = worst_overall
res["tolerance_m"] = 1e-6
res["PASS"] = bool(
    worst_overall < 1e-6
    and all(s["n_intersection_disagreements"] == 0 for s in res["sites"].values())
)
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(
    f"\n[PARITY] overall max |py - c++| = {worst_overall:.3e} m "
    f"(tolerance 1e-6) -> {'PASS' if res['PASS'] else 'FAIL'}"
)
print(f"[PARITY] -> {OUT}.json")
print("PARITY DONE", flush=True)
