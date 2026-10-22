"""Roofline of the policy forward on one RTX A5000.

Measures the two roofs on this card rather than quoting a datasheet, then places the
17500 policy's batch-1 forward against them. Everything is CUDA-event timed, warm.

Protocol
  matmul peak   bf16 8192^3, 10 warm-up + 50 timed, FLOPs = 2 N^3 per call
  bandwidth     4 GiB bf16 device-to-device copy, 10 warm-up + 50 timed,
                bytes = 2 x tensor (one read + one write)
  forward       the harness's own loader (experiments.robot.robot_utils.get_model via
                server.build_policy's config) and BatchedPolicy.forward, batch 1,
                10 warm-up + 30 timed, on one synthetic observation of the real shapes
  flops         torch.utils.flop_counter.FlopCounterMode over one forward, separately
  profile       torch.profiler over 3 forwards, kernels bucketed GEMM / attention / other

Author: Varun Chandra
Env: learn-gpu. GPU 1 only.
"""

import json
import os
import sys
import time

sys.path.insert(0, "/data_new/vchandr2/rollout_stubs")
sys.path.insert(0, "/data_new/vchandr2/scratch/openvla-oft")
sys.path.insert(0, "/data_new/vchandr2/fastvla-eval")

import numpy as np  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402  stub-finder
import torch  # noqa: E402

OUT = os.environ.get("ROOFLINE_OUT", "/data_new/vchandr2/fastvla-eval/fastvla/server/out")
os.makedirs(OUT, exist_ok=True)
R = {}


def ev(fn, warmup, iters):
    """CUDA-event time a callable. Returns mean seconds over `iters`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / iters


# ---------------------------------------------------------------- 1. card
p = torch.cuda.get_device_properties(0)
R["card"] = dict(
    name=p.name,
    total_memory_bytes=p.total_memory,
    sm_count=p.multi_processor_count,
    capability=f"{p.major}.{p.minor}",
    torch=torch.__version__,
    cuda=torch.version.cuda,
)
print(
    f"[card] {p.name}  {p.total_memory / 2**30:.1f} GiB  {p.multi_processor_count} SMs  "
    f"sm_{p.major}{p.minor}  torch {torch.__version__} cu{torch.version.cuda}",
    flush=True,
)


# ---------------------------------------------------------------- 2a. matmul peak
def matmul_peak(n=8192, warmup=10, iters=50):
    """bf16 n^3 square matmul. Tensors stay local so nothing is deleted out from under
    the timed closure."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    t = ev(lambda: torch.matmul(a, b), warmup, iters)
    return n, t, (2.0 * n**3) / t / 1e12


N, t_mm, peak_tflops = matmul_peak()
R["matmul"] = dict(
    n=N,
    dtype="bfloat16",
    warmup=10,
    iters=50,
    seconds_per_call=t_mm,
    tflops=peak_tflops,
)
print(
    f"[peak] bf16 {N}^3 matmul: {t_mm * 1e3:.3f} ms/call -> {peak_tflops:.2f} TFLOP/s", flush=True
)
torch.cuda.empty_cache()


# ---------------------------------------------------------------- 2b. bandwidth
def copy_bandwidth(gib=4, warmup=10, iters=50):
    """Device-to-device copy of a bf16 tensor. Bytes moved = read + write."""
    src = torch.empty((gib * 2**30) // 2, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    t = ev(lambda: dst.copy_(src), warmup, iters)
    moved = 2 * src.numel() * src.element_size()
    return moved, t, moved / t / 1e9


moved, t_bw, bw = copy_bandwidth()
R["bandwidth"] = dict(
    bytes_per_copy=moved,
    warmup=10,
    iters=50,
    seconds_per_copy=t_bw,
    gb_per_s=bw,
)
print(
    f"[peak] d2d copy {moved / 2**30:.1f} GiB moved: {t_bw * 1e3:.3f} ms -> {bw:.1f} GB/s",
    flush=True,
)
torch.cuda.empty_cache()


# ---------------------------------------------------------------- 3. the forward
# run_libero_eval imports `libero.libero.benchmark` at module load, purely for the task
# registry. This measurement never builds an environment, and the LIBERO-Safety checkout was
# removed in the October cleanup, so the package is stubbed the same way rollout_bootstrap
# stubs the RLDS pipeline: a module whose attributes are real, subclassable classes. The
# model-loading path below is untouched and is the server's own.
import types  # noqa: E402  stub-first


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {})


class _StubFinder:
    """Resolve anything under `libero.` to a stub module, at any depth."""

    def find_module(self, fullname, path=None):
        return self if fullname == "libero" or fullname.startswith("libero.") else None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        m = _Stub(fullname)
        m.__path__ = []
        m.__file__ = "<roofline-stub>"
        m.__loader__ = self
        sys.modules[fullname] = m
        return m


sys.meta_path.insert(0, _StubFinder())

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402  stubbed-libero
from experiments.robot.openvla_utils import (  # noqa: E402  sys.path
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_model  # noqa: E402  sys.path
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK  # noqa: E402  sys.path

from fastvla.server.batched_forward import BatchedPolicy  # noqa: E402  sys.path

BASE = (
    "openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--libero_safety--warmstart-combined--headinit"
)
CKPT = os.environ.get("CKPT") or f"/data_new/vchandr2/runs/{BASE}--17500_chkpt"

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
t0 = time.perf_counter()
model = get_model(cfg)
pp = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
ah = get_action_head(cfg, model.llm_dim)
proc = get_processor(cfg)
cfg.unnorm_key = "libero_safety"
pol = BatchedPolicy(cfg, model, proc, ah, pp)
load_s = time.perf_counter() - t0
print(f"[load] merged-direct from {CKPT}  ({load_s:.1f} s)", flush=True)

# weights actually resident, counted off the live modules
wb = 0
for m in (model, pp, ah):
    wb += sum(p.numel() * p.element_size() for p in m.parameters())
    wb += sum(x.numel() * x.element_size() for x in m.buffers())
R["weights"] = dict(
    bytes=wb, gib=wb / 2**30, params=sum(p.numel() for m in (model, pp, ah) for p in m.parameters())
)
print(
    f"[load] resident weights {wb / 2**30:.2f} GiB, {R['weights']['params'] / 1e9:.2f} B params",
    flush=True,
)

# one synthetic observation of the real shapes -- FLOPs do not depend on pixel values
rng = np.random.RandomState(0)
img = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
wrist = rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)
state = rng.uniform(-1, 1, 8).astype(np.float64)
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
prepared = pol.prepare_request(img, wrist, state, TASK)
n_prompt = int(prepared["input_ids"].shape[1])
R["input"] = dict(
    images=2,
    image_hw=[int(prepared["pixel_values"].shape[-2]), int(prepared["pixel_values"].shape[-1])],
    pixel_values_shape=list(prepared["pixel_values"].shape),
    prompt_tokens=n_prompt,
    proprio_dim=8,
    action_positions=int(NUM_ACTIONS_CHUNK * ACTION_DIM),
    task_label=TASK,
)
print(
    f"[input] pixel_values {list(prepared['pixel_values'].shape)}  prompt {n_prompt} tok  "
    f"action positions {NUM_ACTIONS_CHUNK * ACTION_DIM}",
    flush=True,
)

t = ev(lambda: pol.forward([prepared]), 10, 30)
R["forward"] = dict(batch=1, warmup=10, iters=30, seconds=t, ms=t * 1e3)
print(f"[fwd] {t * 1e3:.2f} ms/call (batch 1, 10 warm-up + 30 timed)", flush=True)

# FLOPs, separately -- the counter adds overhead
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402  measurement-only

# BatchedPolicy.forward is decorated @torch.inference_mode(). FlopCounterMode is a
# TorchDispatchMode and sees nothing under inference mode -- it returns 0. Call the
# undecorated function under no_grad instead; same arithmetic, same shapes.
_raw = getattr(pol.forward, "__wrapped__", None)
fc = FlopCounterMode(display=False)
if _raw is not None:
    with torch.no_grad(), fc:
        _raw(pol, [prepared])
else:
    with fc:
        pol.forward([prepared])
flops = fc.get_total_flops()
assert flops > 0, "FlopCounterMode counted nothing -- check the inference_mode unwrap"
R["flops"] = dict(
    total=int(flops),
    gflops=flops / 1e9,
    counted_via="unwrapped forward under no_grad" if _raw else "decorated forward",
)
print(f"[flops] {flops / 1e9:.2f} GFLOP per forward", flush=True)

# profiler: GEMM vs attention vs everything else, and launch count
from torch.autograd import DeviceType  # noqa: E402  measurement-only
from torch.profiler import ProfilerActivity, profile  # noqa: E402  measurement-only

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        pol.forward([prepared])
    torch.cuda.synchronize()

GEMM = (
    "gemm",
    "gemv",
    "cutlass",
    "sgemm",
    "dot_kernel",
    "splitk",
    "s16816",
    "s1688",
    "nn_64x64",
    "cublas",
    "wgrad",
    "implicit_gemm",
    "ampere_",
    "turing_",
)
ATTN = ("attention", "flash", "mha", "softmax", "scaled_dot")
buckets = {"gemm": 0.0, "attention": 0.0, "other": 0.0}
launches = 0
rows = []
for e in prof.key_averages():
    # key_averages() carries BOTH the ATen op (aten::mm, device_type CPU, with the CUDA
    # time of the kernels it launched attributed to it) and the kernels themselves
    # (device_type CUDA). Summing both double-counts. Keep only the kernels -- and do it
    # by device_type, not by name: the biggest kernel on this model is
    # `void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm...>`, so any filter that
    # rejects "::" throws away 70% of the GPU time.
    if e.device_type != DeviceType.CUDA:
        continue
    cu = getattr(e, "self_cuda_time_total", 0) or getattr(e, "self_device_time_total", 0)
    if cu <= 0:
        continue
    launches += e.count
    nm = e.key.lower()
    # Attention FIRST: the flash kernels are built on cutlass and carry `cutlass` in their
    # mangled template names, so testing GEMM first files every attention kernel as a GEMM
    # and reports attention as 0.0%.
    k = (
        "attention"
        if any(x in nm for x in ATTN)
        else ("gemm" if any(g in nm for g in GEMM) else "other")
    )
    buckets[k] += cu
    rows.append((e.key, k, cu / 1e3 / 3, e.count / 3))
tot = sum(buckets.values()) or 1.0
R["profile"] = dict(
    forwards=3,
    gpu_ms_per_forward=tot / 1e3 / 3,
    gemm_frac=buckets["gemm"] / tot,
    attention_frac=buckets["attention"] / tot,
    other_frac=buckets["other"] / tot,
    kernel_launches_per_forward=launches / 3,
    top=[
        dict(kernel=k[:70], bucket=b, ms=round(ms, 3), calls=round(c, 1))
        for k, b, ms, c in sorted(rows, key=lambda r: -r[2])[:12]
    ],
)
R["profile"]["gpu_busy_frac_of_wall"] = (tot / 1e3 / 3) / R["forward"]["ms"]
print(
    f"[prof] GPU {tot / 1e3 / 3:.2f} ms/fwd of {R['forward']['ms']:.2f} ms wall "
    f"({100 * R['profile']['gpu_busy_frac_of_wall']:.1f}% busy) | "
    f"GEMM {100 * buckets['gemm'] / tot:.1f}%  attn {100 * buckets['attention'] / tot:.1f}%  "
    f"other {100 * buckets['other'] / tot:.1f}%  | {launches / 3:.0f} kernel launches/fwd",
    flush=True,
)

# ---------------------------------------------------------------- 4. arithmetic
ach = flops / R["forward"]["seconds"] / 1e12
frac = ach / peak_tflops
ai = flops / wb
ridge = peak_tflops * 1e12 / (bw * 1e9)
R["roofline"] = dict(
    achieved_tflops=ach,
    frac_of_measured_peak=frac,
    bytes_per_forward=wb,
    bytes_note="resident weights read once; profiler gives no DRAM counter",
    arithmetic_intensity_flops_per_byte=ai,
    ridge_point_flops_per_byte=ridge,
    bound="compute" if ai > ridge else "memory",
    roof_tflops=min(peak_tflops, bw * 1e9 * ai / 1e12),
)
R["roofline"]["frac_of_roof"] = ach / R["roofline"]["roof_tflops"]
_busy = R["profile"]["gpu_ms_per_forward"] / 1e3
R["roofline"]["achieved_tflops_gpu_busy"] = flops / _busy / 1e12
R["roofline"]["frac_of_peak_gpu_busy"] = (flops / _busy / 1e12) / peak_tflops
print(f"[roof] achieved {ach:.2f} TFLOP/s = {100 * frac:.1f}% of measured matmul peak", flush=True)
print(
    f"[roof] AI {ai:.2f} FLOP/byte vs ridge {ridge:.2f} -> {R['roofline']['bound']}-bound; "
    f"roof here {R['roofline']['roof_tflops']:.2f} TFLOP/s, "
    f"at {100 * R['roofline']['frac_of_roof']:.1f}% of it",
    flush=True,
)

with open(os.path.join(OUT, "roofline.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=str)
print("ROOFLINE_DONE", flush=True)
