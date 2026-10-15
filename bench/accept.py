"""M4 acceptance: compare a 60-start run against the M3 N=16 bit-exact reference.

Acceptance rule (set in M4, learned from M3):
  * collisions == 29
  * completions in {25, 26}
  * ZERO collision-verdict flips, per start, against the M3 N=16 reference
  * report max per-start clearance difference and the transition matrix

The reference is bench/runs/ctrl60_n16b1, which M3 measured as bit-identical to M2's
single-process run (max |clearance diff| 0.0000 cm over all 60 starts). Comparing against
it rather than against an aggregate count is the point: M3's delay sweep produced a run
with the right totals and four flipped collision verdicts that happened to cancel.

usage: accept.py <run_dir> [label]
"""

import json
import os
import sys

import numpy as np

ROOT = "/data_new/vchandr2/fastvla-eval"
REFERENCE = os.path.join(ROOT, "bench/runs/ctrl60_n16b1/merged.json")


def load_rows(path):
    with open(path) as fh:
        m = json.load(fh)
    return m, {(r["tag"], r["init_idx"]): r for r in m["rows"]}


def check(run_dir, label=None):
    label = label or os.path.basename(run_dir.rstrip("/"))
    _ref_m, ref = load_rows(REFERENCE)
    m, got = load_rows(os.path.join(run_dir, "merged.json"))
    keys = sorted(ref, key=lambda k: (k[0], k[1]))
    missing = [k for k in keys if k not in got]
    coll = sum(bool(got[k]["collision"]) for k in keys if k in got)
    comp = sum(bool(got[k]["completion"]) for k in keys if k in got)
    coll_flips = [
        f"{t}_i{i}"
        for t, i in keys
        if (t, i) in got and bool(ref[(t, i)]["collision"]) != bool(got[(t, i)]["collision"])
    ]
    comp_flips = [
        f"{t}_i{i}"
        for t, i in keys
        if (t, i) in got and bool(ref[(t, i)]["completion"]) != bool(got[(t, i)]["completion"])
    ]
    cats = {}
    for k in keys:
        if k not in got:
            continue
        a, b = ref[k], got[k]
        key = (
            bool(a["collision"]),
            bool(a["completion"]),
            bool(b["collision"]),
            bool(b["completion"]),
        )
        cats[key] = cats.get(key, 0) + 1
    diag = sum(n for (ca, pa, cb, pb), n in cats.items() if (ca, pa) == (cb, pb))
    d = np.array(
        [abs(ref[k]["min_clearance_cm"] - got[k]["min_clearance_cm"]) for k in keys if k in got]
    )
    res = dict(
        label=label,
        run_dir=run_dir,
        n_rows=m["n_rows"],
        missing=[f"{t}_i{i}" for t, i in missing],
        collisions=coll,
        completions=comp,
        rollouts_per_hour=m["rollouts_per_hour"],
        wall_sec=m["wall_sec_rollouts"],
        total_steps=m["total_steps"],
        collision_flips=coll_flips,
        completion_flips=comp_flips,
        n_diagonal=diag,
        n_compared=int(len(d)),
        max_clearance_diff_cm=float(d.max()) if len(d) else None,
        mean_clearance_diff_cm=float(d.mean()) if len(d) else None,
        bit_identical=bool(len(d) and d.max() == 0.0),
        config=m.get("config"),
        server=m.get("server"),
    )
    res["PASS"] = bool(not missing and coll == 29 and comp in (25, 26) and not coll_flips)
    print(f"=== acceptance: {label} ===")
    print(
        f"  rollouts            {m['n_rows']}/60"
        + (f"  MISSING {res['missing']}" if missing else "")
    )
    print(f"  collisions          {coll}  (required 29)")
    print(f"  completions         {comp}  (required 25 or 26)")
    print(f"  collision flips     {len(coll_flips)}  {coll_flips if coll_flips else ''}")
    print(f"  completion flips    {len(comp_flips)}  {comp_flips if comp_flips else ''}")
    print(f"  transition matrix   {diag}/{len(d)} diagonal")
    print(
        f"  max |clearance diff| {res['max_clearance_diff_cm']:.4f} cm"
        f"{'  (bit-identical)' if res['bit_identical'] else ''}"
    )
    print(
        f"  rollouts/hour       {m['rollouts_per_hour']:.1f}   wall {m['wall_sec_rollouts']:.1f} s"
    )
    print(f"  VERDICT             {'PASS' if res['PASS'] else 'REJECT'}")
    return res


if __name__ == "__main__":
    run = sys.argv[1]
    lab = sys.argv[2] if len(sys.argv) > 2 else None
    r = check(run if os.path.isabs(run) else os.path.join(ROOT, "bench/runs", run), lab)
    out = os.path.join(ROOT, "bench/out", f"m4_accept_{r['label']}.json")
    with open(out, "w") as fh:
        json.dump(r, fh, indent=2)
    print(f"  -> {out}")
    sys.exit(0 if r["PASS"] else 3)
