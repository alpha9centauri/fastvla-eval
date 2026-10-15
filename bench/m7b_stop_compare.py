"""What stopping does: MONITOR=stop against MONITOR=off, per start.

Prints the 2x2 collision transition matrix (off verdict x stop verdict) and the same for
completion, plus every start whose verdict moved. Being a brake and not an avoidance, the
expected shape is collisions prevented and completions lost -- the matrix shows both at
once instead of letting two aggregate counts cancel, which is the mistake the M4
acceptance rule exists to catch.

usage: m7b_stop_compare.py <off_run_dir> <stop_run_dir> <out.json>
"""

import json
import os
import sys


def rows(run_dir):
    with open(os.path.join(run_dir, "merged.json")) as fh:
        return {r["key"]: r for r in json.load(fh)["rows"]}


def matrix(off, stop, field):
    """counts[off_verdict][stop_verdict] over the starts both runs share."""
    c = {"TT": [], "TF": [], "FT": [], "FF": []}
    for k in sorted(set(off) & set(stop)):
        a = "T" if off[k].get(field) else "F"
        b = "T" if stop[k].get(field) else "F"
        c[a + b].append(k)
    return c


def show(name, c):
    n = {k: len(v) for k, v in c.items()}
    print(f"  {name:<12} off=yes -> stop=yes {n['TT']:>3}   off=yes -> stop=no  {n['TF']:>3}")
    print(f"  {'':<12} off=no  -> stop=yes {n['FT']:>3}   off=no  -> stop=no  {n['FF']:>3}")
    return n


def main():
    off_dir, stop_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    off, stop = rows(off_dir), rows(stop_dir)
    shared = sorted(set(off) & set(stop))
    res = dict(
        off_run=off_dir,
        stop_run=stop_dir,
        n_starts=len(shared),
        missing_from_stop=sorted(set(off) - set(stop)),
        off_collisions=sum(bool(off[k]["collision"]) for k in shared),
        stop_collisions=sum(bool(stop[k]["collision"]) for k in shared),
        off_completions=sum(bool(off[k]["completion"]) for k in shared),
        stop_completions=sum(bool(stop[k]["completion"]) for k in shared),
    )
    print(
        f"[STOP] {len(shared)} starts; collisions {res['off_collisions']} -> "
        f"{res['stop_collisions']}, completions {res['off_completions']} -> "
        f"{res['stop_completions']}"
    )
    for field in ("collision", "completion"):
        c = matrix(off, stop, field)
        res[f"{field}_matrix"] = {k: len(v) for k, v in c.items()}
        res[f"{field}_matrix_keys"] = c
        show(field, c)
    held = [k for k in shared if stop[k].get("monitor_held")]
    res["n_held"] = len(held)
    res["held_keys"] = held
    res["held_but_still_collided"] = [k for k in held if stop[k]["collision"]]
    res["collided_without_being_held"] = [
        k for k in shared if stop[k]["collision"] and not stop[k].get("monitor_held")
    ]
    print(
        f"[STOP] braked {len(held)}/{len(shared)} rollouts; "
        f"{len(res['held_but_still_collided'])} braked and still collided, "
        f"{len(res['collided_without_being_held'])} collided with no brake"
    )
    print(f"[STOP] prevented: {', '.join(res['collision_matrix_keys']['TF']) or 'none'}")
    print(f"[STOP] newly colliding: {', '.join(res['collision_matrix_keys']['FT']) or 'none'}")
    print(f"[STOP] completions lost: {', '.join(res['completion_matrix_keys']['TF']) or 'none'}")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[STOP] -> {out}")


if __name__ == "__main__":
    main()
