"""Table 1: end-to-end numbers from the Table 1 run log, with the validity gate.

Acceptance (set by the plan, section 12, and the handoff): 29/60 collisions, and
completions in {25, 26} because oa6_L1 init 6 is a completion coin flip on identical
inputs. Anything else means the audit is invalid.

Env: LOG (the run log), DMON (nvidia-smi dmon samples), OUT
"""

import contextlib
import json
import os
import re
import sys

import numpy as np

LOG = os.environ["LOG"]
DMON = os.environ.get("DMON", "")
OUT = os.environ["OUT"]

rows = []
with open(LOG, errors="replace") as fh:
    for line in fh:
        if line.startswith("[ROLLOUT] {"):
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line[len("[ROLLOUT] ") :]))
with open(LOG, errors="replace") as fh:
    txt = fh.read()

m = re.search(r"\[INFO\] done in (\d+)s", txt)
rollout_wall = int(m.group(1)) if m else None

wrapper = None
for p in ("DONE", "FAILED"):
    if os.path.exists(p):
        with open(p) as fh:
            wrapper = fh.read().strip()
        mm = re.search(r"wall=(\d+)h(\d+)m(\d+)s", wrapper)
        if mm:
            total_wall = int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + int(mm.group(3))
            break
else:
    total_wall = None

n = len(rows)
coll = sum(1 for r in rows if r.get("collision"))
comp = sum(1 for r in rows if r.get("completion"))
steps = np.array([int(r.get("steps", 0)) for r in rows], dtype=np.int64)

res = dict(
    tag="MEASURED",
    log=LOG,
    n_rollouts=n,
    collisions=coll,
    completions=comp,
    reference_collisions=29,
    reference_completions_allowed=[25, 26],
    total_steps=int(steps.sum()),
    steps_mean=float(steps.mean()) if n else None,
    steps_min=int(steps.min()) if n else None,
    steps_max=int(steps.max()) if n else None,
    n_at_300_cap=int((steps >= 300).sum()),
    rollout_loop_wall_sec=rollout_wall,
    wrapper_total_wall_sec=total_wall,
    wrapper_marker=wrapper,
)
if rollout_wall:
    res["rollouts_per_hour_loop"] = 3600.0 * n / rollout_wall
    res["steps_per_sec_loop"] = float(steps.sum()) / rollout_wall
    res["sec_per_rollout_loop"] = rollout_wall / n
    res["ms_per_step_loop"] = 1000.0 * rollout_wall / float(steps.sum())
if total_wall:
    res["rollouts_per_hour_incl_startup"] = 3600.0 * n / total_wall
    res["model_load_and_startup_sec"] = total_wall - (rollout_wall or 0)

if DMON and os.path.exists(DMON):
    sm = {0: [], 1: []}
    mem = {0: [], 1: []}
    with open(DMON, errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            f = line.split()
            if len(f) < 6:
                continue
            try:
                gpu, sm_v, mem_v = int(f[1]), int(f[2]), int(f[3])
            except ValueError:
                continue
            if gpu in sm:
                sm[gpu].append(sm_v)
                mem[gpu].append(mem_v)
    for g in (0, 1):
        if sm[g]:
            a = np.array(sm[g])
            res[f"gpu{g}_sm_util_pct_mean"] = float(a.mean())
            res[f"gpu{g}_sm_util_pct_p50"] = float(np.percentile(a, 50))
            res[f"gpu{g}_sm_util_pct_p95"] = float(np.percentile(a, 95))
            res[f"gpu{g}_mem_util_pct_mean"] = float(np.array(mem[g]).mean())
            res[f"gpu{g}_n_samples"] = int(len(a))

per_task = {}
for r in rows:
    t = r.get("tag")
    d = per_task.setdefault(t, dict(n=0, collisions=0, completions=0, steps=0))
    d["n"] += 1
    d["collisions"] += bool(r.get("collision"))
    d["completions"] += bool(r.get("completion"))
    d["steps"] += int(r.get("steps", 0))
res["per_task"] = per_task
res["per_start"] = [
    dict(
        tag=r["tag"],
        init=r["init_idx"],
        collision=bool(r["collision"]),
        completion=bool(r["completion"]),
        steps=int(r["steps"]),
        min_clearance_cm=r.get("min_clearance_cm"),
    )
    for r in rows
]

valid = n == 60 and coll == 29 and comp in (25, 26)
res["VALID"] = bool(valid)
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
np.savez_compressed(OUT + ".npz", steps=steps)

for k, v in res.items():
    if k not in ("per_start", "per_task"):
        print(f"[T1] {k}: {v}")
print(f"[T1] per task: {json.dumps(per_task)}")
print(f"[T1] -> {OUT}.json")
if not valid:
    print(
        f"[T1] INVALID: got {n} rollouts, {coll} collisions, {comp} completions; "
        f"need 60, 29, 25 or 26. STOP.",
        flush=True,
    )
    sys.exit(3)
print("[T1] VALID: matches the GPU 1 reference.", flush=True)
print("T1 SUMMARY DONE", flush=True)
