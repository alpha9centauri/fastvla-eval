"""M3 step 0: split the monitor's non-clearance cost, and check parity of a restructure.

`_on_substep` runs per MuJoCo substep (25 per control step) and does, in Python:
  * `_carried_now(data)`  -- a full loop over data.ncon building `data.contact[i]` wrappers
  * build `actor_of`      -- ~33 dict inserts for the robot geoms, plus carried
  * a second full loop over data.ncon, again building a wrapper per contact
  * `_update_clearance`   -- ported to C++ in M2, 1.30 ms/step

MODE=split   time the parts (wraps `_carried_now` and `_update_clearance`; the contact
             loop and actor_of build are inline, so they come out as the residual)
MODE=parity  run the reference and the restructured monitor over the same saved states and
             compare every record field, including the order of the `pairs` list

Env: ACTIONS_NPZ, OUT, MODE, [N_CALLS, CONTACTS=0|1]
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
):
    sys.path.insert(0, p)

import rollout_bootstrap  # noqa: F401, E402

import numpy as np  # noqa: E402

import physics_safety as ps  # noqa: E402
from fastvla.labeler import contacts as contacts_mod  # noqa: E402
from fastvla.labeler import patch  # noqa: E402
from fastvla.labeler.replay import parse_key, replay_rollout  # noqa: E402

ACTIONS_NPZ = os.environ["ACTIONS_NPZ"]
OUT = os.environ["OUT"]
MODE = os.environ.get("MODE", "split")
N_CALLS = int(os.environ.get("N_CALLS", "1000"))
DISTMAX, CE = 0.5, 25

acts_all = dict(np.load(ACTIONS_NPZ, allow_pickle=True))


# Parity coverage matters more than count here. The contact-accounting branch only runs
# when a contact involves the DECLARED obstacle, and oa1 has 0 collisions in the reference
# 60-start run -- so a selection of the first three sorted keys (all oa1) would compare 300
# records without ever entering the branch being restructured. These tasks are the ones
# that collide (oa2 9/10, oa3 6/10, oa5 9/10 in M0 Table 1), plus oa1 and oa6_L1 for the
# no-contact path.
PARITY_KEYS = [
    "oa2__i00__base",
    "oa3__i00__base",
    "oa5__i00__base",
    "oa2__i01__base",
    "oa5__i01__base",
    "oa1__i00__base",
    "oa6_L1__i00__base",
]


def pick_rollouts(limit_calls, prefer=None):
    out = []
    for k in prefer or PARITY_KEYS:
        a = acts_all.get(k)
        if a is not None and a.ndim == 2 and len(a):
            out.append((k, a))
    if out:
        return out
    for k in sorted(acts_all):
        a = acts_all[k]
        if a.ndim == 2 and len(a):
            out.append((k, a))
        if len(out) >= 3:
            break
    return out


if MODE == "split":
    patch.install(which="cpp", level="batch")
    if os.environ.get("CONTACTS", "0") == "1":
        contacts_mod.install()
    T = defaultdict(float)
    C = Counter()
    PER = []
    _cur = {"b": None}

    def wrap(name, fn):
        def w(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                if _cur["b"] is not None:
                    _cur["b"][name] += dt
                T[name] += dt
                C[name] += 1

        return w

    ps.SafetyMonitor._carried_now = wrap("carried_now", ps.SafetyMonitor._carried_now)
    ps.SafetyMonitor._update_clearance = wrap("clearance", ps.SafetyMonitor._update_clearance)
    _os = ps.SafetyMonitor._on_substep

    def timed_on_substep(self):
        _cur["b"] = defaultdict(float)
        t0 = time.perf_counter()
        try:
            return _os(self)
        finally:
            b = _cur["b"]
            b["total"] = time.perf_counter() - t0
            PER.append(dict(b))
            _cur["b"] = None

    ps.SafetyMonitor._on_substep = timed_on_substep

    n_sub = 0
    for k, a in pick_rollouts(N_CALLS, prefer=sorted(acts_all)[:3]):
        tag, ii = parse_key(k)
        replay_rollout(
            tag,
            ii,
            a,
            distmax=DISTMAX,
            clearance_every=CE,
            on_step=lambda i: len(PER) < N_CALLS * CE,
        )  # noqa: B023
        n_sub = len(PER)
        print(f"[C0] {k}: {n_sub} substeps", flush=True)
        if n_sub >= N_CALLS * CE:
            break
    keys = sorted({x for d in PER for x in d})
    arr = np.array([[d.get(x, 0.0) for x in keys] for d in PER], dtype=np.float64)
    idx = {x: i for i, x in enumerate(keys)}
    col = lambda x: arr[:, idx[x]] if x in idx else np.zeros(len(arr))  # noqa: E731
    tot = col("total")
    resid = tot - col("carried_now") - col("clearance")
    n_steps = len(PER) / CE
    res = dict(
        mode="split",
        contacts_restructured=os.environ.get("CONTACTS", "0") == "1",
        n_substeps=len(PER),
        n_control_steps=n_steps,
        ms_per_substep={x: float(1000 * col(x).mean()) for x in keys},
        ms_per_control_step={
            "on_substep_total": float(1000 * tot.sum() / n_steps),
            "carried_now": float(1000 * col("carried_now").sum() / n_steps),
            "clearance": float(1000 * col("clearance").sum() / n_steps),
            "contact_loop_and_actor_of_residual": float(1000 * resid.sum() / n_steps),
        },
    )
    print(f"\n[C0] {len(PER)} substeps = {n_steps:.0f} control steps")
    for k2, v in res["ms_per_control_step"].items():
        print(f"  {k2:42s} {v:8.3f} ms/control step")
    np.savez_compressed(OUT + ".npz", components=np.array(keys), per_substep=arr)
    with open(OUT + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[C0] -> {OUT}.json")

else:  # parity

    def collect(use_restructured):
        if use_restructured:
            contacts_mod.install()
        else:
            contacts_mod.uninstall()
        recs = []
        for k, a in pick_rollouts(N_CALLS):
            tag, ii = parse_key(k)
            n, mon = replay_rollout(tag, ii, a, distmax=DISTMAX, clearance_every=CE, max_steps=100)
            recs.append((k, mon.records))
        return recs

    patch.install(which="cpp", level="batch")
    ref = collect(False)
    new = collect(True)
    contacts_mod.uninstall()

    def norm(r):
        """Every field the monitor records, in order, as comparable plain data."""
        return dict(
            t=r["t"],
            contact=r["contact"],
            contact_substep=r["contact_substep"],
            substeps=r["substeps"],
            carried=list(r["carried"]),
            min_clearance=r["min_clearance"],
            closest=r["closest_obstacle"],
            hit=list(r["obstacles_hit"]),
            pairs=[
                (
                    p["substep"],
                    p["actor"],
                    p["actor_geom"],
                    p["obstacle"],
                    p["obstacle_geom"],
                    p["dist"],
                    p["force"],
                )
                for p in r["pairs"]
            ],
            per_obstacle={
                o: (
                    d["contact"],
                    d["max_force"],
                    d["min_clearance"],
                    d["actor"],
                    d["actor_geom"],
                    d["obstacle_geom"],
                )
                for o, d in r["per_obstacle"].items()
            },
        )

    n_cmp, mism = 0, []
    for (k1, r1), (k2, r2) in zip(ref, new, strict=True):
        assert k1 == k2
        if len(r1) != len(r2):
            mism.append(dict(key=k1, why=f"record count {len(r1)} vs {len(r2)}"))
            continue
        for a, b in zip(r1, r2, strict=True):
            n_cmp += 1
            na, nb = norm(a), norm(b)
            if na != nb:
                diff = {f: (na[f], nb[f]) for f in na if na[f] != nb[f]}
                mism.append(dict(key=k1, t=a["t"], fields=list(diff), detail=str(diff)[:400]))
    # coverage: a parity pass that never entered the contact branch proves nothing
    n_contact = sum(1 for _k, rs in ref for r in rs if r["contact"])
    n_pairs = sum(len(r["pairs"]) for _k, rs in ref for r in rs)
    n_carried = sum(1 for _k, rs in ref for r in rs if r["carried"])
    res = dict(
        mode="parity",
        n_records_compared=n_cmp,
        n_mismatches=len(mism),
        mismatches=mism[:20],
        keys=[k for k, _ in ref],
        coverage=dict(
            records_with_contact=n_contact,
            total_pairs_recorded=n_pairs,
            records_with_carried_object=n_carried,
        ),
        BIT_EXACT=bool(not mism and n_contact > 0 and n_pairs > 0),
    )
    with open(OUT + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(
        f"\n[C0] compared {n_cmp} monitor records field for field, including the pairs-list order"
    )
    print(f"[C0] mismatches: {len(mism)}")
    for m in mism[:5]:
        print(f"  {m}")
    print(
        f"[C0] coverage: {n_contact} records with an obstacle contact, "
        f"{n_pairs} contact pairs recorded, {n_carried} records with a carried object"
    )
    print(f"[C0] BIT EXACT (and the contact branch was exercised): {res['BIT_EXACT']}")
    print(f"[C0] -> {OUT}.json")
print("CONTACTS BENCH DONE", flush=True)
