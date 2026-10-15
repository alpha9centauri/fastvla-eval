"""Addendum to Table 2: split the derived "OSC controller + obs assembly" row.

Table 2 timed `MjSim.render` at 1.21 ms per control step for two 256x256 cameras and
flagged it as suspiciously cheap, with the suspicion that robosuite does part of the
render work elsewhere and it was landing in the derived row. This resolves that.

robosuite's MujocoEnv.step runs, per control step:

    for i in range(control_timestep / model_timestep):   # 25 substeps
        sim.forward()                 <- extra forward kinematics, 25x
        _pre_action(action)           <- OSC controller, 25x
        sim.step()                    <- physics (Table 2 "sim"), 25x
        _update_observables()         <- sensor update incl. cameras, 25x
    _post_action(action)              <- reward / done, 1x
    _get_observations()               <- assemble the obs dict, 1x

So the derived row is those five things. Each is timed separately here, plus
`MjSim.render` nested inside `_update_observables`, over 100 control steps.

Env: OUT, N_STEPS (default 100), plus the driver's own variables.
"""

import json
import os
import runpy
import sys
import time
from collections import Counter, defaultdict

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402

OUT = os.environ["OUT"]
N_STEPS = int(os.environ.get("N_STEPS", "100"))

T = defaultdict(float)
C = Counter()
STEPS = []
_cur = {"b": None}
# sim.forward() is called both by robosuite's substep loop and from inside the OSC
# controller's own update (55 calls per control step against 25 physics steps), so a flat
# timer double-counts it against _pre_action. This depth flag splits it by caller, which is
# what makes the decomposition add up to env.step exactly.
_in_osc = {"d": 0}


def _b():
    if _cur["b"] is None:
        _cur["b"] = defaultdict(float)
    return _cur["b"]


def _wrap_osc(name, fn):
    def w(*a, **kw):
        b = _b()
        _in_osc["d"] += 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            _in_osc["d"] -= 1
            b[name] += dt
            T[name] += dt
            C[name] += 1

    return w


def _wrap_split_by_caller(base, fn):
    def w(*a, **kw):
        name = f"{base}_in_osc" if _in_osc["d"] else f"{base}_loop"
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


def _wrap(name, fn, new_bucket=False):
    def w(*a, **kw):
        if new_bucket and _cur["b"] is not None:
            STEPS.append(dict(_cur["b"]))
            _cur["b"] = defaultdict(float)
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


def install():
    import robosuite.utils.binding_utils as bu
    from experiments.robot.libero import run_libero_eval as rle
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.environments.base import MujocoEnv

    # new bucket at the top of each control-step iteration in the driver's loop
    rle.prepare_observation = _wrap("prep", rle.prepare_observation, new_bucket=True)

    OffScreenRenderEnv.step = _wrap("env_step", OffScreenRenderEnv.step)
    # RobotEnv overrides _pre_action (MujocoEnv's is shadowed), so patch the class that
    # actually runs: the OSC controller call is inside it.
    from robosuite.environments.robot_env import RobotEnv

    RobotEnv._pre_action = _wrap_osc("osc_pre_action", RobotEnv._pre_action)
    MujocoEnv._post_action = _wrap("post_action", MujocoEnv._post_action)
    MujocoEnv._update_observables = _wrap("update_observables", MujocoEnv._update_observables)
    MujocoEnv._get_observations = _wrap("get_observations", MujocoEnv._get_observations)
    bu.MjSim.step = _wrap("sim_step", bu.MjSim.step)
    bu.MjSim.forward = _wrap_split_by_caller("sim_forward", bu.MjSim.forward)
    bu.MjSim.render = _wrap("render", bu.MjSim.render)

    import physics_safety as ps

    ps.SafetyMonitor._on_substep = _wrap("monitor", ps.SafetyMonitor._on_substep)
    return "ok"


install()
runpy.run_path(os.path.join(DISP, "rollout_openvla_disp.py"), run_name="__main__")
if _cur["b"] is not None:
    STEPS.append(dict(_cur["b"]))

real = [s for s in STEPS if s.get("env_step", 0) > 0][:N_STEPS]
n = len(real)
keys = sorted({k for s in real for k in s})
arr = np.array([[s.get(k, 0.0) for k in keys] for s in real], dtype=np.float64)
idx = {k: i for i, k in enumerate(keys)}


def col(k):
    return arr[:, idx[k]] if k in idx else np.zeros(n)


env_step = col("env_step")
nested = (
    col("sim_step")
    + col("sim_forward")
    + col("osc_pre_action")
    + col("update_observables")
    + col("post_action")
    + col("get_observations")
)
res = dict(
    n_control_steps=n,
    calls=dict(C),
    ms_per_step={k: float(1000 * col(k).mean()) for k in keys},
    ms_per_step_p95={k: float(1000 * np.percentile(col(k), 95)) for k in keys},
    env_step_ms=float(1000 * env_step.mean()),
    nested_sum_ms=float(1000 * nested.mean()),
    env_step_unaccounted_ms=float(1000 * (env_step - nested).mean()),
    render_calls_per_step=C["render"] / max(1, n),
    update_observables_calls_per_step=C["update_observables"] / max(1, n),
)
np.savez_compressed(OUT + ".npz", components=np.array(keys), per_step=arr)
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)

print(f"\n[T2b] {n} control steps")
print(f"{'component':40s} {'ms/step':>9s} {'p95':>9s} {'calls/step':>11s}")
order = [
    "env_step",
    "sim_step",
    "monitor",
    "sim_forward",
    "osc_pre_action",
    "update_observables",
    "render",
    "get_observations",
    "post_action",
    "prep",
]
for k in order:
    if k in idx:
        print(
            f"{k:40s} {1000 * col(k).mean():9.3f} "
            f"{1000 * np.percentile(col(k), 95):9.3f} {C[k] / max(1, n):11.2f}"
        )
print(
    f"\n[T2b] env_step total {res['env_step_ms']:.2f} ms/step; nested sum "
    f"{res['nested_sum_ms']:.2f} ms; unaccounted inside env_step "
    f"{res['env_step_unaccounted_ms']:.2f} ms"
)
print(f"[T2b] -> {OUT}.json")
print("T2B RENDER SPLIT DONE", flush=True)
