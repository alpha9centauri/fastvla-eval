# M4: policy forward optimisation

Baseline is M3 at the accepted configuration: **542.1 rollouts/hour**, N=8, B=1, GPU 85%,
forward 184.7 ms per call (server-side), queue wait 803.8 ms per request.

Acceptance for every change, carried over from M3: run the 60-start set at N=8, B=1;
collisions must be 29, completions 25 or 26, and **zero collision-verdict flips against
the M3 N=16 reference, per start**. `bench/accept.py` enforces it and was validated both
ways first — it passes the reference against itself and rejects M3's delay-sweep D=10 run,
which had the right totals and four flipped collision verdicts that happened to cancel.

**Measurement noise floor.** The microbenchmark (50 real requests at B=1, fixed fixture)
was run three times on the unchanged baseline: forward 167.97, 167.32, 169.05 ms. So the
floor is **±1 ms, ±0.6%**, and anything inside that is not a result. Stated up front
because most of what follows lands there.

## Step 1: the forward profile

`bench/fwd_run.sh profile baseline` and `… time baseline`, 50 real re-plan observations
captured across all six tasks (`bench/out/m4_requests.npz`, 180 requests, 5 distinct
instructions so prompt lengths span 24-32 tokens).

### Per phase, B=1, CUDA events

| phase | ms/call | share of forward |
|---|---|---|
| **LLM trunk** (`language_model`) | **130.2** | **77.5%** |
| **vision backbone** | **33.9** | **20.2%** |
| projector | 0.84 | 0.5% |
| proprio projector | 0.14 | 0.1% |
| forward total (wall, B=1) | **168.0** | |
| `prepare_request` (CPU: crop, normalise, tokenise) | 11.3 | — |

### Dtypes: everything is already bf16

| module | dtype | parameters |
|---|---|---|
| vision_backbone | bfloat16 | 730,911,680 |
| language_model | bfloat16 | 6,738,939,904 |
| projector | bfloat16 | 71,385,600 |
| proprio_projector | bfloat16 | 16,818,176 |
| action_head | bfloat16 | 151,117,831 |

**Nothing is fp32.** All 7.71 B parameters across every submodule are bfloat16, so step 2c
has no work in it. (TF32 is irrelevant here: `matmul.allow_tf32` is False and the GEMMs are
bf16 tensorop already.)

### Top kernels

Self-CUDA over 50 forwards, 15.7 s total:

| share | kernel |
|---|---|
| 34.7% | `cutlass_80_tensorop_bf16_s16816gemm_bf16_*` |
| 33.0% | `aten::mm` |
| 5.1% | `aten::addmm` |
| 3.0% | `cudaLaunchKernel` (138,450 calls = **2,769 launches per forward**) |
| 2.6% | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_*` |
| 2.5% | `aten::mul` |
| 1.5% | `aten::_flash_attention_forward` |
| 1.4% | `aten::copy_` |

**67.7% of device time is bf16 tensorop GEMM.** Flash attention is already in use. That is
what sets the order of everything below: the forward is compute-bound in matrix multiply
at the precision the hardware is fastest at, so the available levers are launch overhead
(3%), wasted computation, and host-side work — not the arithmetic itself.

### What the sequence actually is

`_build_multimodal_attention` builds `[BOS, 512 vision patches, 1 proprio, prompt-1,
57 action tokens]` ≈ 601 tokens. Two facts follow, and they decide steps 2e and 2f:
the only shape variation is the prompt (24-32 tokens), and **the vision patches come
before the prompt**.

## Step 2: per-change results

Microbenchmark first; the 60-start acceptance only for changes that win it.

| | change | prepare ms | forward ms | vs baseline | microbench | acceptance | kept |
|---|---|---|---|---|---|---|---|
| — | baseline (M3) | 11.3 | **168.0** | — | — | 29/25, 0 flips | **yes** |
| 0 | `cudnn.deterministic` + deterministic algorithms | — | — | — | — | 29/25, 0 flips, **526.2 roll/h (−2.9%)** | no |
| 2a | pre-tokenised prompts, cached per task | **12.3** | 168.5 | +1.0 prepare | **lose** | not run | no |
| 2b | image normalise on the GPU | **7.8** | 168.3 | −3.6 prepare, forward flat | no GPU gain | not run | no |
| 2c | bf16 end to end | — | — | — | **already bf16** | — | n/a |
| 2d | `torch.compile(mode="default")` | 11.8 | **156.7** | **−11.3 (−6.7%)** | **win** | **30/23, 3 flips, 459.0 roll/h (−15%)** | **no** |
| 2d | `torch.compile(mode="reduce-overhead")` | 11.7 | 166.4 | −1.6 (within noise) | lose | not run | no |
| 2e | CUDA graphs, one per prompt length | — | — | — | **capture fails** | not run | no |
| 2f | KV/prefix cache of the prompt | — | — | — | **impossible** | not run | no |
| + | skip the unused LM head | 13.9 | 168.1 | +0.1 (within noise) | no gain | not run | no |

### 2a — pre-tokenised prompts: rejected, and the first attempt was a non-change

The first implementation cached the token ids but still called the combined
`processor(prompt, image)` and overwrote the ids afterwards, so no work was removed and it
measured as exactly nothing (prepare 11.33 ms against 11.34 ms). Reimplemented to route
images through `processor.image_processor` alone so the tokeniser never runs on the hot
path — verified byte-identical ids, mask and pixel values — and it is **slower**: prepare
11.3 → 12.3 ms. Tokenising ~30 tokens is cheaper than paying two separate processor calls.
The premise was wrong: tokenisation was never a cost.

### 2b — GPU image preprocessing: rejected, saves the wrong resource

Implemented from the processor's own `tvf_normalize_params` (the backbone is fused, so one
image becomes six channels normalised twice with different mean/std). It does what it
says: **prepare 11.3 → 7.8 ms, −32%**. It is rejected anyway, for two measured reasons.

1. **It cannot help throughput.** The forward is unchanged (168.3 ms, within noise), and
   the server's CPU preprocessing is already off the critical path: two threads, the
   receiver preprocessing while the forward thread occupies the GPU. At N=8 the server
   serves ~5.4 requests/s, so preprocessing costs 5.4 × 11.3 ms = 61 ms/s — 6% of one core
   — against a forward thread that keeps the GPU 85% busy. Removing 3.6 ms from a resource
   with 94% headroom buys nothing.
2. **It is not numerically identical**: max |difference| in pixel values 7.8e-03 against
   the CPU path (bf16 rounding of the normalise), so it would perturb actions and risk
   exactly the collision flips the acceptance rule exists to catch.

The worker-side saving the step asked about would need the protocol to carry raw 256x256
images so the resize moves too. That is the wrong direction for the same reason as (1), and
more strongly: the worker's observation preprocessing is 7.91 ms/step (AUDIT.md addendum)
on a box with 64 physical cores, and the GPU is the saturated resource. Moving work from
the surplus resource onto the bottleneck cannot raise throughput.

### 2d — torch.compile: the microbenchmark lied, and this is why acceptance exists

`mode="default"` is the only change in M4 that won its microbenchmark, and won it clearly:
forward **168.0 → 156.7 ms, −6.7%**, eleven times the noise floor, with the LLM trunk
130.2 → 119.3 ms. End to end it **lost 15%**: 459.0 rollouts/hour against 542.1.

The cause is the shape variation the sequence analysis predicted. The microbenchmark replays
50 requests and lets Inductor settle; a real run meets five prompt lengths interleaved
across eight workers, and pays recompilation and guard overhead repeatedly during the run.

It also failed correctness outright: **3 collision flips** (oa3 inits 4, 6, 7) and 4
completion flips, 53/60 diagonal, max clearance difference 5.29 cm. Compilation changes
fusion and therefore numerics. Either reason alone rejects it.

`mode="reduce-overhead"` was worse than default (166.4 ms, inside the noise floor) and was
not taken further. That is itself informative for 2e, since reduce-overhead *is* CUDA
graphs.

### 2e — CUDA graphs: capture fails on this trunk

One graph per prompt length, no padding — chosen because M3 measured that padding this
model is numerically wrong (`position_ids=None`, so left-padding shifts RoPE positions and
costs up to 3.2e-01 in actions). Lazy capture: run eagerly once, then capture against
static buffers.

Capture does not work:

```
[GRAPH] capture failed for (1, 603): CUDA error: operation failed due to a previous error during capture
[GRAPH] capture failed for (1, 601): captures_underway == 0 INTERNAL ASSERT FAILED at
        "../c10/cuda/CUDACachingAllocator.cpp":2650
```

The implementation falls back to eager and the measurement is therefore the baseline. The
ceiling was small anyway and is worth stating: launch overhead is 3.0% of device time
(2,769 launches per forward), so a perfect graph could recover ~5 ms of 168 ms, and
`reduce-overhead` — which uses CUDA graphs inside Inductor — already delivered nothing
measurable. Not pursued further.

### 2f — KV/prefix caching: structurally impossible, not merely unprofitable

The step was conditional on the prompt prefill being a meaningful share. It is not, and
more decisively the cache cannot exist: the sequence is `[BOS, vision patches, prompt,
action tokens]`, so the prompt tokens attend to the vision patches that precede them, and
those patches are different at every re-plan step. The prompt's keys and values therefore
change every step by construction. The only prefix-stable token is BOS — 1 of ~601.

Even ignoring that, the prompt is 24-32 of ~601 tokens, about 5%, so the upper bound would
have been ~5% of the trunk.

### The unused LM head: a real waste, too small to matter

The profile prompted a check: the L1-regression path reads only
`hidden_states[-1]` and never `.logits`, yet `LlamaForCausalLM` runs `lm_head` over every
position — about 601 x 4096 x 32k x 2 = 158 GFLOP per forward, discarded. Calling the inner
`LlamaModel` instead removes it and is **bit-exact by construction**, since it deletes a
computation whose result was dropped.

Measured: 168.1 ms against 168.0. At the A5000's bf16 tensorop rate 158 GFLOP is ~1.3 ms
theoretical, i.e. 0.8% — below the ±1 ms noise floor. Correct, free, and not worth keeping
under a rule that says a change must win on throughput. Recorded because the reasoning
("the model computes something it throws away") is the kind that usually does pay, and here
the arithmetic says why it does not.

## Step 3: where M4 leaves the forward

**No change was accepted.** The headline configuration is unchanged from M3: N=8, B=1, no
server optimisations.

The honest summary is that this forward is already close to its floor on this card. It is
entirely bf16, it already uses flash attention, 67.7% of its device time is in cuBLAS/cutlass
bf16 tensorop GEMM, and the three classes of lever available to M4 each failed for a
different measured reason: host-side work is off the critical path (2a, 2b), launch overhead
is 3% and the tooling that would capture it either fails (2e) or regresses (2d
reduce-overhead), and the one wasted computation is 0.8% (LM head). The single change that
did speed up the forward in isolation (`torch.compile`) made the system slower because the
workload has five shapes, and broke per-start verdicts.

What would actually move it is outside M4's scope: fewer vision patches (a model change),
a second GPU (the plan's optional M8), or a smaller policy.

## Step 3: the headline run

`RUN_LOG_NAME=M4_T1`, 60 starts, N=8, B=1, D=0, no server optimisations
(`bench/out/m4_t1_summary.json`, `bench/out/m4_accept_M4_T1.json`).

| metric | M3 | **M4_T1** |
|---|---|---|
| **rollouts/hour** | 542.1 | **544.8** |
| wall-clock, 60 starts | 398.5 s | 396.5 s |
| collisions / completions | 29 / 25 | **29 / 25** |
| collision flips vs reference | 0 | **0** |
| transition matrix | 60/60 diagonal | **60/60 diagonal** |
| max per-start clearance difference | — | 0.1530 cm |
| **GPU 1 utilisation** | 85.0% | **85.4%** |
| control step | — | 197.8 ms (8 steps per request) |

544.8 against 542.1 is the same number — two runs of the same configuration — which is the
correct outcome given that nothing was accepted.

### Per-request decomposition, 2,039 requests across 8 workers

| | ms | share |
|---|---|---|
| round trip, worker side | 1008.2 (p95 1196.9) | 100% |
| server queue wait | 798.1 | **79.2%** |
| GPU forward | 184.2 | 18.3% |
| transport + server preprocessing | 25.9 | 2.6% |

Queue wait is 79% of the round trip, and that is the shape of a saturated GPU rather than a
defect: eight workers share one card, so a request waits behind other workers' forwards.
It costs nothing in results, because a worker blocks and its environment does not advance
while it waits. It does mean the system is throughput-limited by the forward: the GPU
serves 1000/184.2 = 5.4 forwards/s, each rollout needs 300/8 = 37.5 of them, giving
5.4 x 3600 / 37.5 = 521 rollouts/hour — within 4% of the 544.8 measured, so the model of
the bottleneck is right.

Which is why M4 produced no gain: every ms off the forward is throughput, and the forward
had no ms to give.
