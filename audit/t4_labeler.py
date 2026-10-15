"""Table 4: microbenchmark of the Python GJK clearance labeler on saved states.

No policy and no GPU compute: the rollout is replayed from the actions Table 1 saved
(`actions_I0_ctrl.npz`), which the driver writes precisely so a rollout can be replayed
without the model. Rendering still happens because OffScreenRenderEnv has no way to turn
cameras off, but render time is outside every timer here.

The scene setup and the replay loop follow rollout_openvla_disp.py's invariants by
importing the same modules it imports -- seed 7+init, env.seed/reset, unwrap_domain,
SafetyMonitor, set_init_state, SceneGeom, attach, then WAIT=10 dummy steps before the
first real action, with the monitor indexed from 0.

Measured per `_update_clearance` call: wall time, and the number of exact
DistanceEngine.distance evaluations it actually performed (= hull pairs that survived
both pruning levels). Swept over distmax, because the plan records a ~543 ms/step figure
"with a wide cutoff (the cutoff defeats the bounding-sphere pruning)" and that claim is
worth a number.

Env: ACTIONS_NPZ, OUT, [N_CALLS=1000, CLEARANCE_EVERY=25, DISTMAX_LIST=0.5,0.1,0.05]
"""
import json
import os
import random
import sys
import time

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP):
    sys.path.insert(0, p)

import rollout_bootstrap  # noqa: F401

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import convex_distance as cd
import geom_shift
import physics_safety as ps
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from physics_safety import SafetyMonitor, unwrap_domain

ACTIONS_NPZ = os.environ["ACTIONS_NPZ"]
OUT = os.environ["OUT"]
N_CALLS = int(os.environ.get("N_CALLS", "1000"))
CLEARANCE_EVERY = int(os.environ.get("CLEARANCE_EVERY", "25"))
DISTMAX_LIST = [float(x) for x in os.environ.get("DISTMAX_LIST", "0.5,0.1,0.05").split(",")]
SEED, WAIT, RES = 7, 10, 256

TASKS = [dict(suite="obstacle_avoidance", level=0, lid=0, tag="oa1"),
         dict(suite="obstacle_avoidance", level=0, lid=1, tag="oa2"),
         dict(suite="obstacle_avoidance", level=0, lid=2, tag="oa3"),
         dict(suite="obstacle_avoidance", level=0, lid=3, tag="oa4"),
         dict(suite="obstacle_avoidance", level=0, lid=4, tag="oa5"),
         dict(suite="obstacle_avoidance", level=1, lid=0, tag="oa6_L1")]
BY_TAG = {t["tag"]: t for t in TASKS}

# ---- instrumentation: count exact pair evaluations, time each clearance call ----------
STATE = dict(pairs=0)
_orig_distance = cd.DistanceEngine.distance
_orig_uc = ps.SafetyMonitor._update_clearance
REC = []       # (wall_sec, n_exact_pairs, min_clearance_m)


def _counting_distance(self, data, g1, g2):
    STATE["pairs"] += 1
    return _orig_distance(self, data, g1, g2)


def _timed_uc(self, dd, actor_of):
    n0 = STATE["pairs"]
    t0 = time.perf_counter()
    try:
        return _orig_uc(self, dd, actor_of)
    finally:
        dt = time.perf_counter() - t0
        rec = self._acc["per_obstacle"] if self._acc else {}
        mc = min([d["min_clearance"] for d in rec.values()], default=float("nan"))
        REC.append((dt, STATE["pairs"] - n0, float(mc)))


cd.DistanceEngine.distance = _counting_distance
ps.SafetyMonitor._update_clearance = _timed_uc

acts_all = dict(np.load(ACTIONS_NPZ, allow_pickle=True))
print(f"[T4] {len(acts_all)} saved action arrays in {ACTIONS_NPZ}", flush=True)
bd = benchmark.get_benchmark_dict()


def replay(tag, ii, acts, distmax, clearance_every, budget):
    """Replay one saved rollout; returns the number of clearance calls recorded."""
    spec = BY_TAG[tag]
    ts = bd[spec["suite"]]()
    task = ts.get_task_by_level_id(spec["level"], spec["lid"])
    init = np.asarray(ts.get_task_init_states(spec["level"], spec["lid"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        f"L{spec['level']}", task.bddl_file)
    seed = SEED + ii
    random.seed(seed)
    np.random.seed(seed)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    env.seed(seed)
    env.reset()
    dom = unwrap_domain(env)
    mon = SafetyMonitor(dom, distmax=distmax, clearance_every=clearance_every)
    env.set_init_state(init[ii])
    geom_shift.SceneGeom(dom, mon)
    mon.attach()
    n0 = len(REC)
    for _ in range(WAIT):
        env.step(get_libero_dummy_action("openvla"))
    for i, a in enumerate(np.asarray(acts, float)):
        mon.begin_step(i)
        env.step(a.tolist())
        mon.end_step()
        if len(REC) - n0 >= budget:
            break
    n_obs_hull_pieces = sum(len(v) for v in mon.obstacle_geoms.values())
    mon.detach()
    env.close()
    return len(REC) - n0, n_obs_hull_pieces


results = {}
for distmax in DISTMAX_LIST:
    REC.clear()
    got, pieces, used = 0, {}, []
    for k in sorted(acts_all):
        if got >= N_CALLS:
            break
        # key is "<tag>__i<NN>__<arm>"
        parts = k.split("__")
        tag, ii = parts[0], int(parts[1][1:])
        a = acts_all[k]
        if a.ndim != 2 or len(a) == 0:
            continue
        n, pc = replay(tag, ii, a, distmax, CLEARANCE_EVERY, N_CALLS - got)
        got += n
        pieces[tag] = pc
        used.append(dict(key=k, calls=n, steps=int(len(a))))
        print(f"[T4] distmax={distmax} {k}: {n} clearance calls "
              f"({got}/{N_CALLS})", flush=True)
    A = np.array(REC, dtype=np.float64)
    ms = A[:, 0] * 1000.0
    pr = A[:, 1]
    results[f"distmax_{distmax}"] = dict(
        distmax=distmax, clearance_every=CLEARANCE_EVERY, n_calls=int(len(A)),
        ms_per_call_mean=float(ms.mean()), ms_per_call_p50=float(np.percentile(ms, 50)),
        ms_per_call_p95=float(np.percentile(ms, 95)), ms_per_call_max=float(ms.max()),
        exact_pairs_per_call_mean=float(pr.mean()),
        exact_pairs_per_call_p95=float(np.percentile(pr, 95)),
        exact_pairs_per_call_max=int(pr.max()),
        us_per_exact_pair=(float(1e6 * A[:, 0].sum() / pr.sum()) if pr.sum() > 0 else None),
        obstacle_hull_pieces=pieces, rollouts_used=used,
        ms_per_control_step=float(ms.mean() * 2.0),   # substep 1 and substep 25
    )
    np.savez_compressed(f"{OUT}_distmax{distmax}.npz", calls=A)
    r = results[f"distmax_{distmax}"]
    print(f"[T4] distmax={distmax}: {r['n_calls']} calls, "
          f"mean {r['ms_per_call_mean']:.2f} ms, p95 {r['ms_per_call_p95']:.2f} ms, "
          f"{r['exact_pairs_per_call_mean']:.1f} exact pairs/call, "
          f"{r['us_per_exact_pair']:.0f} us/pair", flush=True)

json.dump(results, open(OUT + ".json", "w"), indent=2)
print(f"[T4] -> {OUT}.json")
print("T4 LABELER DONE", flush=True)
