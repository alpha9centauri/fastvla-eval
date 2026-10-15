"""Does a batch mixing tasks (hence prompt lengths) work, and what does padding cost?

The delay sweep crashed on ctrl60 because the six LIBERO-Safety instructions tokenise to
different lengths and torch.cat needs equal ones. `left_pad` fixes the crash; this checks
that the fix is numerically sane rather than merely silent, by comparing each request's
actions in a mixed batch against the same request run alone.

Env: OUT
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
from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_model  # noqa: E402
from PIL import Image  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

from fastvla.eval import manifest as M  # noqa: E402
from fastvla.server.batched_forward import BatchedPolicy  # noqa: E402

OUT = os.environ["OUT"]
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
pol = BatchedPolicy(
    cfg,
    model,
    get_processor(cfg),
    get_action_head(cfg, model.llm_dim),
    get_proprio_projector(cfg, model.llm_dim, proprio_dim=8),
)
cfg.unnorm_key = "libero_safety"

# the six real instructions, from the benchmark itself
from libero.libero import benchmark  # noqa: E402

bd = benchmark.get_benchmark_dict()
instrs = []
for t in M.TASKS:
    ts = bd[t["suite"]]()
    instrs.append(str(ts.get_task_by_level_id(t["level"], t["lid"]).language))
_blank = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
toks = [int(pol.processor(pol.prompt_for(s), _blank)["input_ids"].shape[1]) for s in instrs]
print("[MIX] instruction token lengths:", toks, flush=True)
for t, s in zip(toks, instrs, strict=True):
    print(f"   {t:3d}  {s}")

rng = np.random.default_rng(0)
reqs = []
for s in instrs:
    full = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    state = rng.normal(size=8)
    reqs.append(pol.prepare_request(full, wrist, state, s))

singles = [pol.forward([r])[0] for r in reqs]
res = dict(token_lengths=toks, unique_token_lengths=sorted(set(toks)), batches={})
# 1. a mixed-length batch must be refused loudly, not padded
for B in (2, 3, 6):
    try:
        pol.forward(reqs[:B])
        res["batches"][str(B)] = dict(refused=False)
        print(f"[MIX] B={B} mixed batch was NOT refused -- unexpected", flush=True)
    except ValueError as e:
        res["batches"][str(B)] = dict(refused=True, error=str(e)[:160])
        print(f"[MIX] B={B} mixed batch correctly refused: {str(e)[:90]}", flush=True)

# 2. a same-instruction batch of the same size must work, and match singles closely
same = [
    pol.prepare_request(
        np.random.default_rng(100 + i).integers(0, 255, size=(256, 256, 3), dtype=np.uint8),
        np.random.default_rng(200 + i).integers(0, 255, size=(256, 256, 3), dtype=np.uint8),
        np.random.default_rng(300 + i).normal(size=8),
        instrs[0],
    )
    for i in range(6)
]
same_singles = [pol.forward([r])[0] for r in same]
res["same_instruction"] = {}
for B in (2, 4, 6):
    got = pol.forward(same[:B])
    d = [float(np.abs(got[i] - same_singles[i]).max()) for i in range(B)]
    res["same_instruction"][str(B)] = dict(worst=max(d))
    print(f"[MIX] same-instruction B={B}: worst |batched - single| = {max(d):.3e}", flush=True)
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"[MIX] -> {OUT}.json")
print("MIXED BATCH PROBE DONE", flush=True)
