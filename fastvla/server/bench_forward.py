"""M4 microbenchmark and profile of the policy forward alone.

Fixed inputs: bench/out/m4_requests.npz, real re-plan observations captured from rollouts
across all six tasks (5 distinct instructions, so prompt lengths vary 24-32 tokens).
Every variant is timed on the same requests in the same order.

MODE=time     per-phase latency at B=1 over N requests, with CUDA events
MODE=profile  torch.profiler over the same requests: per-kernel top 20, self CUDA/CPU
MODE=dtypes   what is bf16 and what is fp32, per submodule

VARIANT selects the change under test (step 2):
  baseline        as M3 shipped it
  pretok          prompts tokenised once per instruction and cached (2a)
  gpu_preproc     image crop/normalise on the GPU as a batched op (2b)
  compile_default torch.compile(mode="default") on the LLM trunk (2d)
  compile_reduce  torch.compile(mode="reduce-overhead") on the LLM trunk (2d)
Variants compose where sensible via VARIANT=pretok+gpu_preproc.

Env: FIXTURE, OUT, MODE, VARIANT, [N_REQ=50, CKPT]
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

import rollout_bootstrap  # noqa: F401, E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_model  # noqa: E402
from fastvla.server.batched_forward import BatchedPolicy  # noqa: E402
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402

FIXTURE = os.environ.get("FIXTURE", "/data_new/vchandr2/fastvla-eval/bench/out/m4_requests.npz")
OUT = os.environ["OUT"]
MODE = os.environ.get("MODE", "time")
VARIANT = os.environ.get("VARIANT", "baseline")
N_REQ = int(os.environ.get("N_REQ", "50"))
BASE = (
    "openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--libero_safety--warmstart-combined--headinit"
)
CKPT = os.environ.get("CKPT", f"/data_new/vchandr2/runs/{BASE}--17500_chkpt")
VARIANTS = set(VARIANT.split("+"))

torch.backends.cudnn.benchmark = os.environ.get("CUDNN_BENCHMARK", "0") == "1"

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
pp = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
ah = get_action_head(cfg, model.llm_dim)
proc = get_processor(cfg)
cfg.unnorm_key = "libero_safety"
pol = BatchedPolicy(cfg, model, proc, ah, pp)

z = np.load(FIXTURE, allow_pickle=True)
FULL, WRIST, STATE = z["full"], z["wrist"], z["state"]
INSTR = z["instruction"].tolist()
META = [json.loads(s) for s in z["meta"]]
n = min(N_REQ, len(FULL))
print(f"[FWD] variant={VARIANT} mode={MODE} requests={n}", flush=True)

# ---------------------------------------------------------------- variants ------------
res_pretok_check = None
res_gpuprep_check = None
applied = []
if "pretok" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_prompt_cache(pol))
    _v = O.verify_prompt_cache(pol, FULL[0], WRIST[0], STATE[0], INSTR[0])
    print(f"[FWD] prompt-cache identity check: {_v}", flush=True)
    res_pretok_check = _v
    if not (_v["input_ids_equal"] and _v["attention_mask_equal"] and _v["pixel_values_equal"]):
        raise SystemExit(f"prompt cache changes the inputs, refusing to measure it: {_v}")
if "gpu_preproc" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_gpu_preprocessing(pol))
    _g = O.verify_gpu_preprocessing(pol, FULL[0], WRIST[0], STATE[0], INSTR[0])
    print(f"[FWD] gpu-preproc identity check: {_g}", flush=True)
    res_gpuprep_check = _g
if "skip_lm_head" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_skip_lm_head(pol))
if "cudagraph" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_skip_lm_head(pol))
    applied.append(O.enable_cuda_graphs(pol))
if "compile_default" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_compile(pol, mode="default"))
if "compile_reduce" in VARIANTS:
    from fastvla.server import optimisations as O

    applied.append(O.enable_compile(pol, mode="reduce-overhead"))
print(f"[FWD] applied: {applied}", flush=True)

res = dict(
    variant=VARIANT,
    mode=MODE,
    n_requests=n,
    applied=applied,
    torch=torch.__version__,
    fixture=FIXTURE,
)

# ---------------------------------------------------------------- dtypes --------------
if MODE == "dtypes":
    groups = {
        "vision_backbone": model.vision_backbone,
        "language_model": model.language_model,
        "projector": getattr(model, "projector", None),
        "proprio_projector": pp,
        "action_head": ah,
    }
    out = {}
    for name, mod in groups.items():
        if mod is None:
            continue
        counts, nelem = {}, {}
        for _pn, prm in mod.named_parameters():
            k = str(prm.dtype)
            counts[k] = counts.get(k, 0) + 1
            nelem[k] = nelem.get(k, 0) + prm.numel()
        out[name] = dict(
            param_tensors_by_dtype=counts,
            params_by_dtype={k: int(v) for k, v in nelem.items()},
            total_params=int(sum(nelem.values())),
        )
    res["dtypes"] = out
    print(f"{'module':22s} {'dtype':18s} {'tensors':>8s} {'params':>14s}")
    for name, d in out.items():
        for dt, c in d["param_tensors_by_dtype"].items():
            print(f"{name:22s} {dt:18s} {c:8d} {d['params_by_dtype'][dt]:14,d}")
    with open(OUT + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[FWD] -> {OUT}.json")
    print("FWD BENCH DONE", flush=True)
    raise SystemExit(0)

# ---------------------------------------------------------------- phase timing --------
PHASES = {}


def instrument_phases():
    """CUDA-event timers on the four big submodules, plus the H2D copy."""
    import torch.nn as nn

    def wrap(name, mod):
        if mod is None or not isinstance(mod, nn.Module):
            return
        orig = mod.forward

        def fwd(*a, **kw):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            try:
                return orig(*a, **kw)
            finally:
                e.record()
                PHASES.setdefault(name, []).append((s, e))

        mod.forward = fwd

    wrap("vision_backbone", model.vision_backbone)
    wrap("language_model", model.language_model)
    wrap("proprio_projector", pp)
    wrap("action_head", ah)
    if getattr(model, "projector", None) is not None:
        wrap("projector", model.projector)


def drain_phases():
    torch.cuda.synchronize()
    out = {}
    for k, evs in PHASES.items():
        out[k] = [s.elapsed_time(e) for s, e in evs]
    PHASES.clear()
    return out


reqs = [(FULL[i], WRIST[i], STATE[i], INSTR[i]) for i in range(n)]

# warm up: weights, allocator, any compile
for i in range(min(5, n)):
    pol.forward([pol.prepare_request(*reqs[i])])
torch.cuda.synchronize()

if MODE == "profile":
    from torch.profiler import ProfilerActivity, profile

    prepared = [pol.prepare_request(*r) for r in reqs]
    instrument_phases()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_modules=True,
    ) as prof:
        for p in prepared:
            pol.forward([p])
        torch.cuda.synchronize()
    ph = drain_phases()
    ka = prof.key_averages()
    with open(OUT + "_kernels_cuda.txt", "w") as fh:
        fh.write(ka.table(sort_by="self_cuda_time_total", row_limit=20))
    with open(OUT + "_kernels_cpu.txt", "w") as fh:
        fh.write(ka.table(sort_by="self_cpu_time_total", row_limit=20))
    tot_cuda = sum(e.self_cuda_time_total for e in ka)
    top = [
        dict(
            name=e.key[:70],
            self_cuda_ms=e.self_cuda_time_total / 1000.0,
            self_cpu_ms=e.self_cpu_time_total / 1000.0,
            calls=e.count,
            share_pct=100.0 * e.self_cuda_time_total / max(1, tot_cuda),
        )
        for e in sorted(ka, key=lambda x: -x.self_cuda_time_total)[:20]
    ]
    res["profile"] = dict(
        self_cuda_ms_total=tot_cuda / 1000.0,
        top20=top,
        phases_ms={
            k: dict(mean=float(np.mean(v)), total=float(np.sum(v)), calls=len(v))
            for k, v in ph.items()
        },
    )
    print(f"\n[FWD] per-phase, CUDA events, {n} forwards at B=1:")
    print(f"{'phase':22s} {'ms/call':>9s} {'calls':>7s} {'share of fwd':>13s}")
    fwd_total = sum(np.sum(v) for v in ph.values())
    for k, v in sorted(ph.items(), key=lambda x: -np.sum(x[1])):
        print(f"{k:22s} {np.mean(v):9.3f} {len(v):7d} {100 * np.sum(v) / fwd_total:12.1f}%")
    print(f"\n[FWD] top kernels by self CUDA (total {tot_cuda / 1000.0:.0f} ms):")
    for t in top[:12]:
        print(
            f"  {t['share_pct']:5.1f}%  {t['self_cuda_ms']:8.1f} ms  "
            f"{t['calls']:6d}x  {t['name'][:62]}"
        )
else:  # MODE=time
    prep_ms, fwd_ms = [], []
    instrument_phases()
    for r in reqs:
        t0 = time.perf_counter()
        p = pol.prepare_request(*r)
        t1 = time.perf_counter()
        pol.forward([p])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        prep_ms.append(1000 * (t1 - t0))
        fwd_ms.append(1000 * (t2 - t1))
    ph = drain_phases()
    a, b = np.array(prep_ms), np.array(fwd_ms)
    res["timing"] = dict(
        prepare_ms_mean=float(a.mean()),
        prepare_ms_p95=float(np.percentile(a, 95)),
        forward_ms_mean=float(b.mean()),
        forward_ms_p95=float(np.percentile(b, 95)),
        total_ms_mean=float((a + b).mean()),
        phases_ms={
            k: dict(mean=float(np.mean(v)), total=float(np.sum(v)), calls=len(v))
            for k, v in ph.items()
        },
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    print(
        f"\n[FWD] {VARIANT}: prepare {a.mean():7.2f} ms   forward {b.mean():7.2f} ms   "
        f"total {(a + b).mean():7.2f} ms   (p95 fwd {np.percentile(b, 95):.2f})"
    )
    for k, v in sorted(ph.items(), key=lambda x: -np.sum(x[1])):
        print(f"       {k:22s} {np.mean(v):8.3f} ms/call x {len(v)}")
    np.savez_compressed(OUT + ".npz", prepare_ms=a, forward_ms=b)

with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"[FWD] -> {OUT}.json")
print("FWD BENCH DONE", flush=True)
