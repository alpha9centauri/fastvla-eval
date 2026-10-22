# Roofline of the policy forward

Author: Varun Chandra. Measured 2026-10-22 on GPU 1, `learn-gpu` environment, under
`audit/run_with_alert.py` with `RUN_LOG_NAME=roofline_17500`. Every number below comes from a
single run of [`roofline.py`](roofline.py); raw output in `out/roofline.json`.

## The card

```
NVIDIA RTX A5000    24564 MiB (23.7 GiB usable)   64 SMs   sm_86
max SM clock 2100 MHz    max memory clock 8001 MHz
torch 2.2.0+cu121, CUDA 12.1
```

## Measured roofs, not datasheet

| roof | protocol | result |
|---|---|---|
| bf16 matmul | 8192³ square, 10 warm-up + 50 timed, CUDA events, FLOPs = 2N³ | **94.19 TFLOP/s** (11.674 ms/call) |
| memory bandwidth | 4 GiB bf16 device-to-device copy, 10 warm-up + 50 timed; bytes = 2× tensor (read + write) | **685.9 GB/s** (12.523 ms for 8.0 GiB moved) |

The A5000's datasheet figure for bf16 tensor-core throughput with FP32 accumulate is 117.5
TFLOP/s; 94.19 is what a plain `torch.matmul` reaches on this card, and it is the number the
forward is measured against. Bandwidth 685.9 GB/s against a 768 GB/s datasheet peak, 89%, which is
normal for a copy kernel. The matmul roof moves about ±1% run to run (92.58 / 93.14 / 93.59 /
94.19 across four runs), so treat 58% of peak as 58 ± 1, not as three significant figures.

## The forward

Loaded merged-direct through the harness's own path — `experiments.robot.robot_utils.get_model`
on the same `GenerateConfig` that `fastvla/server/server.py:build_policy` constructs, then
`BatchedPolicy.forward`. Checkpoint `…--headinit--17500_chkpt`, 7.71 B parameters, **14.39 GiB**
resident across model, proprio projector and action head.

One synthetic observation of the real shapes, since FLOPs do not depend on pixel values:

```
pixel_values   [1, 12, 224, 224]    two 224x224 RGB views, 6 channels each (two vision towers)
prompt         34 tokens            "pick up the black bowl between the plate and the ramekin..."
proprio        8-dim
action         56 positions         NUM_ACTIONS_CHUNK 8 x ACTION_DIM 7
```

| quantity | value |
|---|---|
| forward, batch 1 | **166.52 ms** (CUDA events, 10 warm-up + 30 timed) |
| FLOPs per forward | **9029.54 GFLOP** (`FlopCounterMode`, one forward) |
| achieved | **54.22 TFLOP/s** |
| fraction of measured matmul peak | **57.6%** |
| bytes per forward | 15.45 GB (resident weights, read once) |
| arithmetic intensity | **584.4 FLOP/byte** |
| ridge point | **137.3 FLOP/byte** (94.19 TFLOP/s ÷ 685.9 GB/s) |
| verdict | **compute-bound**, by a factor of 4.3 in intensity |

## Where the GPU time goes

`torch.profiler` over 3 forwards, bucketed by kernel. Only `device_type == CUDA` entries are
counted: `key_averages()` also carries the ATen op rows (`aten::mm`) with their kernels' time
attributed to them, and summing both double-counts.

| bucket | share of GPU time | ms/forward |
|---|---|---|
| GEMM | **79.7%** | 119.7 |
| attention (flash) | 3.2% | 4.9 |
| everything else | 17.0% | 25.5 |
| **GPU busy** | | **150.13 ms of 166.52 ms wall — 90.2%** |
| kernel launches | | **2948 per forward** |

Top kernels: `cutlass_80_tensorop_bf16_s16816gemm` ~109 ms over 330 calls, then
`ampere_bf16_s16816gemm_…` ~8 ms over 204, then a long tail of elementwise and `CatArrayBatchedCopy`.

## Reading

The forward is **compute-bound and not close to the roof**: 584 FLOP/byte against a ridge of 137
puts it 4.3x into the compute regime, so bandwidth is not the constraint and no amount of
weight-streaming cleverness helps — but it runs at 54.22 TFLOP/s against a 94.19 TFLOP/s roof, 57.6%.
The gap is three separate things and only one of them is interesting. First, **9.8% of the wall
is GPU idle** (16.39 ms per forward): 2948 kernel launches at batch 1 leave gaps the GPU cannot
fill, which is the ordinary small-batch launch-overhead tax and the thing CUDA graphs or batching
would attack. Second, **17.0% of the busy time is non-GEMM** (25.5 ms of elementwise, concatenation
and normalisation) which carries almost no FLOPs and therefore runs at a tiny fraction of peak;
that is the fusion opportunity. Third, and the real ceiling, **the GEMMs themselves reach only
75.4 TFLOP/s, 80.1% of the square-matmul peak** — unsurprising, because at batch 1 with ~600 token
positions the matrices are skinny rather than square, and a skinny GEMM cannot saturate the
tensor cores the way an 8192³ one does. Taken together: the floor at this batch size is about
95.9 ms (9029.54 GFLOP at 94.19 TFLOP/s) against 166.52 ms measured, so roughly 71 ms is
recoverable in principle, perhaps 47 ms of it (the idle plus the non-GEMM) by engineering that
does not change the arithmetic, and the rest only by making the GEMMs less skinny — which means
larger batches, which is exactly what the server's batching path is for. The measurement is also
consistent with `FORWARD.md`'s earlier per-phase split, where the LLM trunk was 77.5% of a
167 ms forward; this says the same thing from the kernel side.

## Reproducing

```bash
cd fastvla-eval/audit
env CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONPATH=/data_new/vchandr2/rollout_stubs:<openvla-oft>:/data_new/vchandr2/fastvla-eval \
    RUN_LOG_NAME=roofline_17500 DISPLACEMENT_TOTAL=1 \
    DISPLACEMENT_CMD="python -u ../fastvla/server/roofline.py" \
    python -u run_with_alert.py --no-wandb
```

Needs the 17500 checkpoint and `openvla-oft` at
`e4287e94541f459edc4feabc4e181f537cd569a8` with
`vla-safety-interp/src/displacement/patches/openvla_oft_lazy_vla_import.patch` applied, plus the
`transformers` fork at `bc339d9ad707454c0c115970db43c260067c61ab` — the pins in
`envs/libero-vla.requirements.txt`. See `envs/BUILD.md`.

Two things `roofline.py` stubs, both inert for this measurement and both documented in the file:
the `libero` package (imported by `run_libero_eval` only for the task registry; no environment is
ever built) and the RLDS pipeline (via the harness's own `rollout_bootstrap`). The model-loading
path itself is the server's, unmodified.

Two measurement traps worth recording, because both produced plausible wrong numbers first:
`FlopCounterMode` is a `TorchDispatchMode` and sees **nothing** under `torch.inference_mode`, so
it returns 0 against `BatchedPolicy.forward` as decorated — the script calls the undecorated
function under `no_grad` instead. And the largest kernel on this model is
`void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm…>`, so bucketing kernels by rejecting
names containing `::` discards 70% of the GPU time; bucket by `device_type` instead. Attention
must also be tested before GEMM, because the flash kernels are built on cutlass and carry
`cutlass` in their mangled names.
