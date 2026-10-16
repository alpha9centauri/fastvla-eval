"""Compare the post-trained checkpoint against the base, per start.

Same shape as bench/m7b_stop_compare.py: aggregate counts first, then the per-start
transition matrix, because two runs can agree on totals and disagree on individual starts.

usage: p3_compare.py <base_run> <posttrained_run> <out.json>
"""

import json
import os
import sys

import numpy as np


def rows(run_dir):
    with open(os.path.join(run_dir, "merged.json")) as fh:
        return {r["key"]: r for r in json.load(fh)["rows"]}


def matrix(a, b, field, keys):
    out = {"TT": [], "TF": [], "FT": [], "FF": []}
    for k in keys:
        out[("T" if a[k].get(field) else "F") + ("T" if b[k].get(field) else "F")].append(k)
    return out


def show(name, c):
    n = {k: len(v) for k, v in c.items()}
    print(f"  {name:<12} base=yes -> ft=yes {n['TT']:>3}   base=yes -> ft=no  {n['TF']:>3}")
    print(f"  {'':<12} base=no  -> ft=yes {n['FT']:>3}   base=no  -> ft=no  {n['FF']:>3}")
    return n


def main():
    base_dir, ft_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    a, b = rows(base_dir), rows(ft_dir)
    keys = sorted(set(a) & set(b))
    res = dict(base_run=base_dir, posttrained_run=ft_dir, n_starts=len(keys))
    for field in ("collision", "completion"):
        res[f"base_{field}s"] = sum(bool(a[k][field]) for k in keys)
        res[f"ft_{field}s"] = sum(bool(b[k][field]) for k in keys)
    print(
        f"[P3] {len(keys)} starts; collisions {res['base_collisions']} -> "
        f"{res['ft_collisions']}, completions {res['base_completions']} -> "
        f"{res['ft_completions']}"
    )
    for field in ("collision", "completion"):
        c = matrix(a, b, field, keys)
        res[f"{field}_matrix"] = {k: len(v) for k, v in c.items()}
        res[f"{field}_matrix_keys"] = c
        show(field, c)
    # how far each run got, and how close it came
    for tag, d in (("base", a), ("ft", b)):
        st = np.array([d[k]["steps"] for k in keys], float)
        cl = np.array([d[k]["min_clearance_cm"] for k in keys], float)
        res[f"{tag}_steps_mean"] = float(st.mean())
        res[f"{tag}_steps_median"] = float(np.median(st))
        res[f"{tag}_min_clearance_cm_mean"] = float(cl.mean())
        print(
            f"  {tag:<4} steps mean {st.mean():6.1f} median {np.median(st):6.1f} | "
            f"min-clearance mean {cl.mean():.3f} cm"
        )
    # of the starts the base collided on, what happened
    base_coll = [k for k in keys if a[k]["collision"]]
    res["base_collided_n"] = len(base_coll)
    res["base_collided_ft_steps_mean"] = float(np.mean([b[k]["steps"] for k in base_coll]))
    res["base_collided_base_steps_mean"] = float(np.mean([a[k]["steps"] for k in base_coll]))
    print(
        f"  on the {len(base_coll)} starts the base collides: steps "
        f"{res['base_collided_base_steps_mean']:.1f} -> "
        f"{res['base_collided_ft_steps_mean']:.1f}"
    )
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[P3] -> {out}")


if __name__ == "__main__":
    main()
