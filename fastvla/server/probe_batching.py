"""M3 step 1 feasibility: is OpenVLA-OFT's predict_action safe at batch > 1?

Two things could break batching, both found by reading
prismatic/extern/hf/modeling_prismatic.py:

  1. predict_action() line ~971 appends the empty-token id 29871 when the prompt does not
     already end with it, using a hard-coded shape (1, 1):
         torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), 0)), dim=1)
     At B > 1 that cat is a shape error -- unless the branch never fires because the
     tokenizer already ends the prompt with 29871.
  2. _run_diffusion_prediction samples noise with size=(1, ...). Not on our path (this
     checkpoint uses the L1 regression head), but it would be a batch-1 assumption.

This answers both empirically, and measures what batching actually buys, by running the
same observations one at a time and then together and comparing the actions elementwise.

Env: OUT, [CKPT]
"""

import json
import os
import sys
import time

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402
import torch  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_model  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

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
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
processor = get_processor(cfg)
cfg.unnorm_key = "libero_safety"

res = {"ckpt": CKPT, "torch": torch.__version__}

# ---- does the prompt already end with 29871? -------------------------------------------
prompt = "In: What action should the robot take to pick up the bottle?\nOut:"
rng = np.random.default_rng(0)
img = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
from PIL import Image  # noqa: E402

inp = processor(prompt, Image.fromarray(img))
last = int(inp["input_ids"][0, -1])
res["prompt_last_token_id"] = last
res["ends_with_29871"] = bool(last == 29871)
print(f"[PROBE] prompt last token id = {last}; ends with 29871: {last == 29871}")
print(
    f"[PROBE] input_ids shape {tuple(inp['input_ids'].shape)}, "
    f"pixel_values shape {tuple(inp['pixel_values'].shape)}"
)
res["input_ids_len"] = int(inp["input_ids"].shape[1])


def build(nimg_seed, proprio_seed):
    """One request's processor output, two images concatenated as the driver does."""
    r = np.random.default_rng(nimg_seed)
    a = Image.fromarray(r.integers(0, 255, size=(224, 224, 3), dtype=np.uint8))
    b = Image.fromarray(r.integers(0, 255, size=(224, 224, 3), dtype=np.uint8))
    ia = processor(prompt, a).to("cuda:0", dtype=torch.bfloat16)
    ib = processor(prompt, b).to("cuda:0", dtype=torch.bfloat16)
    ia["pixel_values"] = torch.cat([ia["pixel_values"], ib["pixel_values"]], dim=1)
    pr = np.random.default_rng(proprio_seed).normal(size=8).astype(np.float64)
    return ia, pr


REQS = [build(1, 11), build(2, 22), build(3, 33), build(4, 44)]


def run_single(req):
    inputs, proprio = req
    with torch.inference_mode():
        act, _ = model.predict_action(
            **dict(inputs),
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=False,
        )
    return np.asarray(act, dtype=np.float64)


EMPTY_TOKEN = 29871


def _append_empty_token(input_ids, attn):
    """Pre-append the empty token so predict_action's batch-1 cat branch never fires.

    predict_action appends id 29871 itself when the prompt does not already end with it,
    with a hard-coded shape (1, 1) -- which is a shape error at B > 1. Doing it here, for
    the whole batch at once, makes `torch.all(input_ids[:, -1] == 29871)` true, so the
    branch is skipped and every later step is batch-shaped. The resulting sequence is
    identical to what the single-request path builds, so parity is unaffected.
    """
    B = input_ids.shape[0]
    tok = torch.full((B, 1), EMPTY_TOKEN, dtype=input_ids.dtype, device=input_ids.device)
    one = torch.ones((B, 1), dtype=attn.dtype, device=attn.device)
    return torch.cat([input_ids, tok], dim=1), torch.cat([attn, one], dim=1)


def run_batch(reqs, preappend=True):
    input_ids = torch.cat([r[0]["input_ids"] for r in reqs], dim=0)
    attn = torch.cat([r[0]["attention_mask"] for r in reqs], dim=0)
    pix = torch.cat([r[0]["pixel_values"] for r in reqs], dim=0)
    proprio = np.stack([r[1] for r in reqs], axis=0)
    if preappend:
        input_ids, attn = _append_empty_token(input_ids, attn)
    with torch.inference_mode():
        act, _ = model.predict_action(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pix,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=False,
        )
    return np.asarray(act, dtype=np.float64)


singles = [run_single(r) for r in REQS]
res["single_action_shape"] = list(singles[0].shape)
print(f"[PROBE] single action shape {singles[0].shape}")

res["batch"] = {}
for B in (2, 4):
    try:
        out = run_batch(REQS[:B])
        print(f"[PROBE] B={B} ok, batched output shape {out.shape}")
        # the batched output should contain each request's chunk
        per = out.reshape(B, -1, singles[0].shape[-1]) if out.ndim == 2 else out
        diffs = [float(np.abs(per[i] - singles[i]).max()) for i in range(B)]
        res["batch"][str(B)] = dict(
            ok=True, shape=list(out.shape), max_abs_diff_vs_single=diffs, worst=max(diffs)
        )
        print(f"[PROBE] B={B} max |batched - single| per request: {[f'{d:.3e}' for d in diffs]}")
    except Exception as e:
        res["batch"][str(B)] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        print(f"[PROBE] B={B} FAILED: {type(e).__name__}: {e}")

# ---- latency vs batch size --------------------------------------------------------------
res["latency_ms"] = {}
for B in (1, 2, 4, 8, 16):
    reqs = [REQS[i % len(REQS)] for i in range(B)]
    try:
        for _ in range(2):
            run_batch(reqs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            run_batch(reqs)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / n
        res["latency_ms"][str(B)] = dict(total_ms=ms, per_request_ms=ms / B)
        print(f"[PROBE] B={B:2d}: {ms:7.2f} ms per forward, {ms / B:6.2f} ms per request")
    except Exception as e:
        res["latency_ms"][str(B)] = dict(error=f"{type(e).__name__}: {e}")
        print(f"[PROBE] B={B} latency FAILED: {type(e).__name__}: {e}")

with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"[PROBE] -> {OUT}.json")
print("PROBE BATCHING DONE", flush=True)
