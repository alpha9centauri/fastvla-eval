"""Capture real re-plan observations as a fixed fixture for the M4 microbenchmark.

Every M4 variant must be timed on identical inputs, so the requests are captured once
from real rollouts and saved. Spanning several tasks matters: the prompt tokenises to 24,
28, 29, 30 or 32 tokens depending on the instruction, and step 2e (CUDA graphs) turns on
exactly that spread.

Observations are taken at re-plan steps only -- the policy runs once per 8-step chunk, so
those are the states it actually sees. Actions come from the saved M3 run so no policy is
needed here and the trajectory is the real one.

Env: OUT, [N_PER_TASK=2, MAX_STEPS=120]
"""

import json
import os
import sys

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    "/data_new/vchandr2/fastvla-eval",
):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402

from fastvla.eval import manifest as M  # noqa: E402

OUT = os.environ["OUT"]
N_PER_TASK = int(os.environ.get("N_PER_TASK", "2"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "120"))
ACTIONS = os.environ.get(
    "ACTIONS_DIR", "/data_new/vchandr2/fastvla-eval/bench/runs/ctrl60_n16b1/arrays"
)
WAIT, RES, CHUNK = 10, 256, 8


def main():
    import random

    from experiments.robot.libero.libero_utils import get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation
    from experiments.robot.robot_utils import get_image_resize_size
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    class _Cfg:
        model_family = "openvla"

    resize_size = get_image_resize_size(_Cfg())
    bd = benchmark.get_benchmark_dict()
    full, wrist, state, instr, meta = [], [], [], [], []

    for t in M.TASKS:
        ts = bd[t["suite"]]()
        task = ts.get_task_by_level_id(t["level"], t["lid"])
        init_states = np.asarray(ts.get_task_init_states(t["level"], t["lid"]))
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, f"L{t['level']}", task.bddl_file
        )
        lang = str(task.language)
        for ii in range(N_PER_TASK):
            key = f"{t['tag']}__i{ii:02d}__base"
            path = os.path.join(ACTIONS, f"{key}.npz")
            if not os.path.exists(path):
                print(f"[CAP] no saved actions for {key}, skipping", flush=True)
                continue
            acts = np.load(path)["actions"]
            seed = 7 + ii
            random.seed(seed)
            np.random.seed(seed)  # noqa: NPY002
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
            try:
                env.seed(seed)
                env.reset()
                obs = env.set_init_state(init_states[ii])
                for _ in range(WAIT):
                    obs, _r, _d, _i = env.step(get_libero_dummy_action("openvla"))
                for s in range(min(MAX_STEPS, len(acts))):
                    if s % CHUNK == 0:
                        o, _img = prepare_observation(obs, resize_size)
                        full.append(np.asarray(o["full_image"], np.uint8))
                        wrist.append(np.asarray(o["wrist_image"], np.uint8))
                        state.append(np.asarray(o["state"], np.float64))
                        instr.append(lang)
                        meta.append(dict(tag=t["tag"], init=ii, policy_step=s))
                    obs, _r, dn, _i = env.step(np.asarray(acts[s], float).tolist())
                    if dn:
                        break
            finally:
                env.close()
            print(f"[CAP] {key}: {len(full)} requests so far", flush=True)

    np.savez_compressed(
        OUT,
        full=np.stack(full),
        wrist=np.stack(wrist),
        state=np.stack(state),
        instruction=np.array(instr, dtype=object),
        meta=np.array([json.dumps(m) for m in meta], dtype=object),
    )
    print(f"[CAP] saved {len(full)} requests -> {OUT}")
    from collections import Counter

    print(f"[CAP] per task: {Counter(m['tag'] for m in meta)}")
    print("CAPTURE DONE", flush=True)


if __name__ == "__main__":
    main()
