r"""Per-component timers for the existing rollout driver, applied by monkeypatch.

The driver (`rollout_openvla_disp.py`) is a regression reference and must not be edited
or copied, so nothing here touches it. Instead the functions and methods it calls are
wrapped BEFORE it is executed with runpy, which works because the driver binds those
names at its own import time -- by then the wrappers are already in place.

Decomposition of one control step. The monitor hooks `sim.step`, so monitor cost is
nested INSIDE env.step, and so is rendering:

    env.step(a)                                   t_env_step
      |- sim.step() x n_substeps                    t_sim      (pure MuJoCo physics)
      |    \- SafetyMonitor._on_substep()           t_monitor  (contacts every substep)
      |         \- _update_clearance()              t_clearance (GJK, every 25th substep)
      |- sim.render() x 2 cameras                   t_render
      \- OSC controller + obs assembly            (derived: env_step - sim - monitor - render)

and outside env.step, per control step:

    prepare_observation()                         t_prep
    get_action()                                  t_forward   (only on re-plan steps, 1 in 8)
    process_action()                              t_proc

Buckets are cut at each prepare_observation call, which is the first thing the driver
does in a control-step iteration. The WAIT=10 dummy steps run before the first
prepare_observation and land in a "warmup" bucket that the summary drops.
"""

import json
import time
from collections import Counter, defaultdict

STEPS = []  # one dict per control step: component -> seconds
CALLS = Counter()  # component -> number of calls
PAIRS = []  # hull pairs per _update_clearance call
_cur = {"b": None}
_prof = {"p": None}  # optional torch.profiler handle, stepped on each forward


def _bucket():
    if _cur["b"] is None:
        _cur["b"] = defaultdict(float)
        _cur["b"]["_warmup"] = 1.0
    return _cur["b"]


def _new_bucket():
    if _cur["b"] is not None:
        STEPS.append(dict(_cur["b"]))
    _cur["b"] = defaultdict(float)


def flush():
    if _cur["b"] is not None:
        STEPS.append(dict(_cur["b"]))
        _cur["b"] = None


def set_profiler(p):
    _prof["p"] = p


def _timed(name, fn, new_bucket=False, count_pairs=False):
    def wrapper(*a, **kw):
        if new_bucket:
            _new_bucket()
        b = _bucket()
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            b[name] += time.perf_counter() - t0
            CALLS[name] += 1

    return wrapper


def install(count_pairs=True):
    """Wrap every component. Returns a dict naming what was patched, for the record.

    count_pairs=False skips the DistanceEngine.distance wrapper. That wrapper is called
    ~200 times per clearance call, and its own cost is charged to the clearance timer, so
    the clean per-step breakdown is taken with it off. Pair counts come from a separate
    run with it on.
    """
    import convex_distance as cd
    import physics_safety as ps
    import robosuite.utils.binding_utils as bu
    from experiments.robot import robot_utils as ru
    from experiments.robot.libero import run_libero_eval as rle
    from libero.libero.envs import OffScreenRenderEnv

    patched = {}

    # --- outside env.step -----------------------------------------------------
    rle.prepare_observation = _timed("prep", rle.prepare_observation, new_bucket=True)
    patched["prepare_observation"] = "experiments.robot.libero.run_libero_eval"
    rle.process_action = _timed("proc", rle.process_action)
    patched["process_action"] = "experiments.robot.libero.run_libero_eval"

    _ga = ru.get_action

    def get_action(*a, **kw):
        b = _bucket()
        t0 = time.perf_counter()
        try:
            return _ga(*a, **kw)
        finally:
            b["forward"] += time.perf_counter() - t0
            CALLS["forward"] += 1
            p = _prof["p"]
            if p is not None:
                p.step()

    ru.get_action = get_action
    patched["get_action"] = "experiments.robot.robot_utils"

    # --- env.step and what nests inside it ------------------------------------
    _base_step = OffScreenRenderEnv.step
    OffScreenRenderEnv.step = _timed("env_step", _base_step)
    patched["OffScreenRenderEnv.step"] = "libero.libero.envs"

    bu.MjSim.step = _timed("sim", bu.MjSim.step)
    patched["MjSim.step"] = "robosuite.utils.binding_utils"
    bu.MjSim.render = _timed("render", bu.MjSim.render)
    patched["MjSim.render"] = "robosuite.utils.binding_utils"

    ps.SafetyMonitor._on_substep = _timed("monitor", ps.SafetyMonitor._on_substep)
    patched["SafetyMonitor._on_substep"] = "physics_safety"

    _uc = ps.SafetyMonitor._update_clearance

    def _update_clearance(self, dd, actor_of):
        b = _bucket()
        n0 = CALLS["gjk_pair"]
        t0 = time.perf_counter()
        try:
            return _uc(self, dd, actor_of)
        finally:
            b["clearance"] += time.perf_counter() - t0
            CALLS["clearance"] += 1
            PAIRS.append(CALLS["gjk_pair"] - n0)

    ps.SafetyMonitor._update_clearance = _update_clearance
    patched["SafetyMonitor._update_clearance"] = "physics_safety"

    # Per-pair exact GJK calls, so "hull pairs per clearance call" is counted, not guessed.
    if count_pairs:
        cd.DistanceEngine.distance = _timed("gjk_pair", cd.DistanceEngine.distance)
        patched["DistanceEngine.distance"] = "convex_distance"
    else:
        patched["DistanceEngine.distance"] = "NOT patched (count_pairs=False)"

    return patched


def save(path, meta):
    flush()
    import numpy as np

    keys = sorted({k for s in STEPS for k in s} - {"_warmup"})
    real = [s for s in STEPS if "_warmup" not in s]
    arr = np.array([[s.get(k, 0.0) for k in keys] for s in real], dtype=np.float64)
    np.savez_compressed(
        path + ".npz",
        components=np.array(keys),
        per_step=arr,
        pairs_per_clearance_call=np.array(PAIRS, dtype=np.int64),
    )
    with open(path + ".json", "w") as f:
        json.dump(
            dict(
                meta=meta,
                components=keys,
                calls=dict(CALLS),
                n_control_steps=len(real),
                n_warmup_buckets=len(STEPS) - len(real),
            ),
            f,
            indent=2,
        )
    return keys, arr
