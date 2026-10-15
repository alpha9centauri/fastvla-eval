"""M3 step 1: does BatchedPolicy reproduce the reference get_action, and what does B buy?

Parity is checked against the reference path itself -- `get_action(...)` from
experiments.robot.robot_utils, the exact call the driver makes -- on real observations
captured from a replayed rollout, not on synthetic images. For each captured observation:

  reference:  get_action(cfg, model, observation, LANG, ...)        -> list of 8 (7,) arrays
  ours, B=1:  BatchedPolicy.forward([prepared])                    -> one (8, 7) array
  ours, B=k:  BatchedPolicy.forward(k prepared requests)            -> k (8, 7) arrays

max |difference| must be 0 for parity mode to be honest about reproducing the reference.

Env: ACTIONS_NPZ, OUT, [N_OBS=16, CKPT]
"""

import json
import os
import sys
import time

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
import torch  # noqa: E402
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig,
    prepare_observation,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

from fastvla.server.batched_forward import BatchedPolicy  # noqa: E402

OUT = os.environ["OUT"]
N_OBS = int(os.environ.get("N_OBS", "16"))
BASE = (
    "openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--libero_safety--warmstart-combined--headinit"
)
CKPT = os.environ.get("CKPT", f"/data_new/vchandr2/runs/{BASE}--17500_chkpt")

cfg = GenerateConfig(
    model_family="openvla",
    pretrained_checkpoint=CKPT,
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    lora_rank=32,
    env_img_res=256,
)
model = get_model(cfg)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
processor = get_processor(cfg)
cfg.unnorm_key = "libero_safety"
resize_size = get_image_resize_size(cfg)

# ---- capture real observations by stepping one env ------------------------------------
import random  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action  # noqa: E402
from experiments.robot.libero.run_libero_eval import process_action  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

bd = benchmark.get_benchmark_dict()
ts = bd["obstacle_avoidance"]()
task = ts.get_task_by_level_id(0, 0)
init = np.asarray(ts.get_task_init_states(0, 0))
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, "L0", task.bddl_file)
seed = 7
random.seed(seed)
np.random.seed(seed)  # noqa: NPY002
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(seed)
env.reset()
obs = env.set_init_state(init[0])
LANG = str(task.language)
for _ in range(10):
    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

OBS = []
from collections import deque  # noqa: E402

q = deque(maxlen=NUM_ACTIONS_CHUNK)
while len(OBS) < N_OBS:
    observation, _ = prepare_observation(obs, resize_size)
    if len(q) == 0:
        OBS.append(
            dict(
                full_image=observation["full_image"].copy(),
                wrist_image=observation["wrist_image"].copy(),
                state=np.asarray(observation["state"]).copy(),
            )
        )
        q.extend(
            get_action(
                cfg,
                model,
                observation,
                LANG,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                use_film=False,
            )
        )
    a = np.asarray(q.popleft(), dtype=np.float64)
    obs, _, dn, _ = env.step(np.asarray(process_action(a, "openvla")).tolist())
    if dn:
        break
env.close()
print(f"[BP] captured {len(OBS)} re-plan observations from oa1 init 0", flush=True)

# ---- reference actions ------------------------------------------------------------------
ref = []
for o in OBS:
    chunk = get_action(
        cfg,
        model,
        dict(full_image=o["full_image"], wrist_image=o["wrist_image"], state=o["state"]),
        LANG,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        use_film=False,
    )
    ref.append(np.asarray(chunk, dtype=np.float64))
ref = np.stack(ref, axis=0)
print(f"[BP] reference actions {ref.shape}", flush=True)

pol = BatchedPolicy(cfg, model, processor, action_head, proprio_projector)
prepared = [pol.prepare_request(o["full_image"], o["wrist_image"], o["state"], LANG) for o in OBS]

res = dict(n_obs=len(OBS), ref_shape=list(ref.shape), batch={}, latency_ms={})
for B in (1, 2, 4, 8):
    try:
        got = []
        for s in range(0, len(prepared), B):
            got.extend(pol.forward(prepared[s : s + B]))
        got = np.stack(got[: len(OBS)], axis=0)
        d = float(np.abs(got - ref).max())
        res["batch"][str(B)] = dict(ok=True, max_abs_diff=d, exact=bool(d == 0.0))
        print(f"[BP] B={B}: max |ours - reference| = {d:.3e}  exact={d == 0.0}", flush=True)
    except Exception as e:
        res["batch"][str(B)] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"[BP] B={B} FAILED: {type(e).__name__}: {e}", flush=True)

for B in (1, 2, 4, 8, 12):
    reqs = [prepared[i % len(prepared)] for i in range(B)]
    try:
        for _ in range(2):
            pol.forward(reqs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            pol.forward(reqs)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / n
        res["latency_ms"][str(B)] = dict(
            forward_ms=ms, per_request_ms=ms / B, throughput_req_s=1000.0 * B / ms
        )
        print(
            f"[BP] B={B:2d}: {ms:7.2f} ms/forward  {ms / B:6.2f} ms/request  "
            f"{1000.0 * B / ms:6.1f} req/s",
            flush=True,
        )
    except Exception as e:
        res["latency_ms"][str(B)] = dict(error=f"{type(e).__name__}: {e}")
        print(f"[BP] B={B} latency FAILED: {type(e).__name__}: {e}", flush=True)
        break

res["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
print(f"[BP] peak torch GPU memory {res['peak_gpu_mem_gb']:.2f} GB")
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"[BP] -> {OUT}.json")
print("BATCHED PARITY DONE", flush=True)
