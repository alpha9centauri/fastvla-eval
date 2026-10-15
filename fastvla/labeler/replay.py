"""Replay a saved rollout so the labeler can be measured without a policy or a GPU.

The scene setup reproduces `rollout_openvla_disp.py`'s invariants exactly -- seed 7+init,
`env.seed()` then `reset()`, `unwrap_domain`, `SafetyMonitor`, `set_init_state`,
`SceneGeom`, `attach()`, then WAIT=10 dummy actions before the first real one, with the
monitor indexed from 0. Those four invariants are what make the numbers comparable to the
driver's; see PROJECT_HANDOFF.md "Rollout-loop invariants".

The actions come from the `ACT_NPZ` the driver writes for exactly this purpose ("executed
actions, so a rollout can be replayed on CPU with no policy and no GPU"), so they are
already `process_action`-ed and are passed straight to `env.step`.

Rendering still happens -- OffScreenRenderEnv has no way to switch cameras off -- but it is
0.7 ms per camera (AUDIT.md addendum) and sits outside every timer here.

This mirrors the replay loop in audit/t4_labeler.py, which produced the M0 Table 4
baseline. It is factored out here because M2 needs the same loop from several benchmarks.
"""
import os
import random

import numpy as np

SEED, WAIT, RES = 7, 10, 256

TASKS = [
    dict(suite="obstacle_avoidance", level=0, lid=0, tag="oa1"),
    dict(suite="obstacle_avoidance", level=0, lid=1, tag="oa2"),
    dict(suite="obstacle_avoidance", level=0, lid=2, tag="oa3"),
    dict(suite="obstacle_avoidance", level=0, lid=3, tag="oa4"),
    dict(suite="obstacle_avoidance", level=0, lid=4, tag="oa5"),
    dict(suite="obstacle_avoidance", level=1, lid=0, tag="oa6_L1"),
]
BY_TAG = {t["tag"]: t for t in TASKS}


def parse_key(k):
    """'oa1__i00__base' -> ('oa1', 0)"""
    parts = k.split("__")
    return parts[0], int(parts[1][1:])


def replay_rollout(tag, ii, actions, distmax=0.5, clearance_every=25, max_steps=None,
                   on_step=None):
    """Replay one saved rollout. Returns (n_control_steps, monitor).

    on_step(i) is called after each control step, so a caller can stop early by raising.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    import geom_shift
    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from physics_safety import SafetyMonitor, unwrap_domain

    spec = BY_TAG[tag]
    bd = benchmark.get_benchmark_dict()
    ts = bd[spec["suite"]]()
    task = ts.get_task_by_level_id(spec["level"], spec["lid"])
    init = np.asarray(ts.get_task_init_states(spec["level"], spec["lid"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        f"L{spec['level']}", task.bddl_file)
    seed = SEED + ii
    random.seed(seed)
    # Legacy global seeding on purpose: the driver seeds exactly this way (line 341) and the
    # replay has to reproduce its scene setup bit for bit.
    np.random.seed(seed)  # noqa: NPY002
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    env.seed(seed)
    env.reset()
    dom = unwrap_domain(env)
    mon = SafetyMonitor(dom, distmax=distmax, clearance_every=clearance_every)
    env.set_init_state(init[ii])
    geom_shift.SceneGeom(dom, mon)
    mon.attach()
    try:
        for _ in range(WAIT):
            env.step(get_libero_dummy_action("openvla"))
        acts = np.asarray(actions, float)
        if max_steps is not None:
            acts = acts[:max_steps]
        n = 0
        for i, a in enumerate(acts):
            mon.begin_step(i)
            env.step(a.tolist())
            mon.end_step()
            n += 1
            if on_step is not None and on_step(i) is False:
                break
        return n, mon
    finally:
        mon.detach()
        env.close()
