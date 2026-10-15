# M3: parallel simulators and a batched policy server

Baseline is M2 on GPU 1: **166.5 rollouts/hour, 80.83 ms per control step, GPU 26.1%**.
Ordering decision taken: M3 then M4, plain PyTorch server first (M3a), then Triton on the
same model (M3b), compared.

## 0. What the measurements already force

Three things were measured before any of this was designed, because they decide the shape
of it. Artefacts: `bench/out/m3_probe_batching.json`, `bench/out/m3_batched_parity.json`.

**The 7B forward is compute-bound at batch 1.** Throughput against batch size, same
observations, GPU 1:

| B | ms per forward | ms per request | requests/s |
|---|---|---|---|
| 1 | 167.14 | 167.14 | 6.0 |
| 2 | 287.38 | 143.69 | 7.0 |
| 4 | 528.31 | 132.08 | **7.6** |
| 8 | 1050.60 | 131.33 | 7.6 |
| 12 | OOM | | |

Batching buys **1.27x** and is flat past B=4. A single request already saturates the card:
~400 tokens of sequence with two 224x224 images' worth of vision patches through a 7B
bf16 model. So dynamic batching is *not* where M3's speedup comes from — parallel
simulators are. That matters for how much engineering each part deserves, and it is a
finding worth reporting rather than hiding, since the plan guessed batching would be
central.

**Usable batch is capped by memory, not by latency.** The weights are ~20 GB of a 23.68 GB
card; peak allocation at B=8 is 21.33 GB and B=12 OOMs. The server caps B at 8 and
defaults to 4. Workers also need EGL render contexts on the same card, so N and B compete
for the same ~2.5 GB of headroom; the scaling sweep records memory, not just throughput.

**Where the ceiling is.** At 7.6 forwards/s and 300/8 = 37.5 forwards per rollout, the GPU
alone caps throughput at about **730 rollouts/hour**. M2's control step is 80.83 ms of
which the forward is 20.9 ms amortised, leaving ~60 ms of CPU work per step that
parallelises across the 64 physical cores. One worker therefore sustains roughly
3600/(300 x 0.060) = 200 rollouts/hour, so the knee should fall near N = 4 and the
expected result is 600-730 rollouts/hour, i.e. 3.6-4.4x over M2. The sweep is what settles
it; this paragraph is the prediction it will be checked against.

## 1. Two batch-1 assumptions in the model code, and how they are handled

Both found by probing, not by reading hopefully.

1. `predict_action` (modeling_prismatic.py:971) appends the empty token id 29871 when the
   prompt does not already end with it, using a hard-coded shape `(1, 1)`. The LIBERO
   prompt ends with 29901 (`:`), so the branch always fires and at B > 1 raises
   *"Sizes of tensors must match except in dimension 1"*. Handled by appending the token
   ourselves for the whole batch (`append_empty_token`), which makes the branch's
   condition true so it is skipped. The sequence built is identical to the single-request
   one.
2. `_regression_or_discrete_prediction` (:924) ends with
   `reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)` = `(8, 7)`, so at B > 1 it raises
   *"shape '[8, 7]' is invalid for input of size 112"*. Handled by rebinding that one
   method on the model class with the same body and `reshape(-1, 8, 7)`
   (`install_reshape_patch`). It is a copy of one function with one line changed, because
   the reshape sits mid-function with no hook around it and the research tree may not be
   edited.

Nothing else in the forward needed changing: it is (B, seq_len, D) throughout, and
`_process_proprio_features` already reshapes proprio to `(bsz, -1)`.

## 2. The request protocol

One request is one re-plan: the policy runs once per 8-step action chunk.

```
request  = { run_id, worker_id, request_id, rollout_key, policy_step,
             instruction, full_image (256x256x3 uint8), wrist_image (256x256x3 uint8),
             state (8,) float64 }
response = { request_id, policy_step, actions (8, 7) float64, server_batch_size,
             queue_wait_ms, forward_ms }
```

- **`request_id`** is `(worker_id, rollout_key, policy_step)` and is unique for the run. The
  worker blocks on its own reply, so correlation is strictly necessary only to detect a
  bug — but it is carried and asserted on, because a silently mismatched reply would look
  like policy nondeterminism and would be diagnosed as the oa6_L1 coin flip.
- **`policy_step`** is the control-step index at which the chunk was requested
  (`t - WAIT`). It is in the protocol for three reasons: it makes a reply verifiable, it
  is what the per-step logs join on, and M4's CUDA-graph work will want fixed-shape
  requests keyed by step parity.
- **Images go as raw uint8**, not as prepared tensors. (Correction, M3b: the figure first
  given here was 2 x 256 x 256 x 3 = 393 KB. The worker actually sends
  `prepare_observation`'s output, which has already been resized to 224x224, so the real
  payload is 2 x 224 x 224 x 3 = **301 KB**. The argument is unchanged -- the processor
  output is 602 KB -- but the number was wrong and the Triton config inherited it, which
  is how it was caught.) The
  processor output is `(1, 6, 224, 224)` bf16 = 602 KB, so sending prepared inputs would
  be 53% more bytes on the wire *and* would hard-code the preprocessing into the protocol.
  The cost is that ~33 ms of per-request CPU preprocessing (crop, normalise, tokenise)
  lands on the server; at the measured 7.6 req/s ceiling that is 25% of one core, and the
  server does it in its receiver thread so it overlaps with the GPU forward.
- **No images come back.** The response is 8 x 7 float64 = 448 bytes plus metadata.

**Transport: ZeroMQ `ROUTER`/`DEALER` over a Unix domain socket (`ipc://`).** Justification:
the server needs to accept N workers on one socket and reply to the right one, which is
exactly what `ROUTER` identities do, with no `.proto` and no service definition; `ipc://`
avoids the loopback TCP stack for 393 KB payloads; and `zmq.Poller` gives the receiver
thread a timeout-driven loop, which is how the batch-delay window D is implemented.
gRPC is available in this env and was considered — it is the plan's suggestion because it
is Triton's native transport — but for M3a it would mean generating stubs to move four
arrays between two processes on one box, and HTTP/2 framing on top. gRPC's real payoff is
M3b, where Triton speaks it anyway; keeping M3a on ZeroMQ makes the M3a-vs-Triton
comparison a comparison of *servers* rather than of transports, and the per-request
overhead of each is reported separately.

## 3. Parity mode

`PARITY=1` sets N = 1, B = 1, D = 0 and asserts all three. In that mode:

- the worker is the only client, so each forward sees exactly one request;
- `BatchedPolicy.forward` on a single request is **bit-exact** against the reference
  `get_action` — verified at max |difference| **0.000e+00** on 16 real re-plan
  observations captured from a replayed rollout (`bench/out/m3_batched_parity.json`);
- the worker's rollout loop reproduces the driver's four invariants (gripper
  `process_action` before `env.step`, WAIT = 10 dummy steps with the monitor indexed from
  `t - WAIT`, completion from the env's `done` flag at the break, and the scene-setup
  order), so the whole rollout must match the reference driver step for step.

Fast mode is *not* bit-exact and is not claimed to be: at B > 1 the bf16 GEMM reduction
order changes, giving max |action difference| **4.8e-3 at B=2 and 6.6e-3 at B=4/8** against
the same inputs. That is the plan's expected Tier-A/Tier-B split, and the number is
reported rather than buried. Collisions and completions are then checked statistically
(Tier B) and per start.

## 4. Writing results from N workers without the A1-A4 bugs

`audit/single_env_assumptions.md` catalogues why running the driver N times would corrupt
its own output. Each is broken deliberately:

| assumption | how M3 avoids it |
|---|---|
| **A1** `_save()` rewrites every output file in full from in-process accumulators, so two processes overwrite rather than merge | A worker never writes a shared file. It writes **one file per rollout**, `results/<run_id>/rollouts/<rollout_key>.json`, created once and never rewritten. The manifest runner merges them at the end into `results/<run_id>/rows.json`. A crashed worker loses at most its current rollout, and the merge is a pure function of the directory. |
| **A2** fixed output names with nothing distinguishing worker or attempt | Every path carries `run_id` (a timestamp plus a short random suffix) and the file is named by `rollout_key` = `<task>__i<init>__<arm>`, which is unique within a run. Arrays go to `results/<run_id>/arrays/<rollout_key>.npz`, one per rollout. |
| **A3** `baseline_<STEP>.npz` read once at import then read-modify-written in the loop | No worker writes a baseline. The unshifted baseline is an *input*; if a run needs one it is read from a path given in the manifest. Nothing in M3 writes to a file it also reads. |
| **A4** videos go to the driver's own directory with no run id, silently overwriting | Videos go to `results/<run_id>/videos/<rollout_key>.mp4`, i.e. under `OUTDIR` and under the run id. Confirmed empirically in M2 that the research tree's `videos/` already contains a name that a re-run would clobber. |
| **A5** one process-global `[PROGRESS] done=N` counter parsed from one child's stdout | Workers report progress on a ZeroMQ `PUSH` socket to the runner, which owns the single `[PROGRESS] done=N total=N` line that `run_with_alert.py` parses. Workers' own stdout is captured per worker to `results/<run_id>/logs/worker<k>.log`. |

Two further rules fall out of M2's experience rather than the list: the runner refuses to
start if `results/<run_id>` exists, and `RUN_LOG_NAME` must be unique per run — M2's
60-start log was truncated when a retry reopened the same name with mode `"w"`, which is
A2 in a different costume.

## 5. The probe hook's `h[0, ...]` assumption (B2)

The driver captures the post-final-norm hidden state with a forward hook that indexes
`h[0, -(N_ACT + 1):-1, :]` — batch element 0 only (driver lines 189, 206-209). At B > 1
that silently returns the first request in the batch for every request, which would not
error and would corrupt a probe trace without any sign.

M3 handles it by **refusing the combination**: `PROBE=1` is rejected unless the server is
in parity mode (B = 1), with the message naming this assumption. In parity mode the hook
is correct as written, because there genuinely is only one request in the batch. Batched
probe capture needs the hook to return per-request slices and the response to carry a
4096-d vector per request; that is a change to what the protocol transports, so it is
deliberately out of M3's scope and recorded here as the reason.

This is also why every M3 measurement runs with `PROBE=0`, as M0, M1 and M2 did.

## 6. Components

```
fastvla/server/batched_forward.py   BatchedPolicy: prepare_request (CPU) + forward (GPU)
fastvla/server/server.py            ZeroMQ ROUTER server, receiver + forward threads
fastvla/server/client.py            worker-side stub; one blocking request/reply
fastvla/sim/worker.py               one env per process, C++ labeler, rollout loop
fastvla/eval/manifest.py            the 60-start and 10-start gate manifests
fastvla/eval/runner.py              spawns server + N workers, merges per-rollout files
fastvla/server/triton/              M3b: the same model as a Triton Python backend
```

The server has two threads on purpose. The receiver thread polls the socket, runs
`prepare_request` for each arrival (the ~33 ms of CPU preprocessing) and pushes onto a
queue; the forward thread pops up to B requests that arrived within D ms and runs one GPU
forward. Preprocessing therefore overlaps with the forward, and the GIL is released for
the duration of the CUDA call.

---

# M3 results

## Step 0: the contact scan

The monitor walked `data.ncon` twice per MuJoCo substep, building a `data.contact[i]`
wrapper per contact each time, 25 substeps per control step. Restructured to read
`dd.contact.geom` and `.dist` as bulk numpy arrays, filter `dist <= CONTACT_DIST_EPS`
vectorised, and visit only surviving contacts in Python in ascending original index order;
`fingers`, the robot half of `actor_of` and the obstacle geom set are built once per
monitor rather than once per substep.

| component | before | after |
|---|---|---|
| SafetyMonitor total | 9.48 ms/step | 3.57 ms/step |
| of which contact scan | 8.19 ms/step | **2.25 ms/step** (3.6x) |
| whole control step | 66.70 ms | 61.52 ms (-7.8%) |

Measured with identical instrumentation on both sides. A first attempt with per-part
timers claimed 17.29 -> 4.04 ms, but that compared unequal instrumentation — the
restructure stops calling `_carried_now`, so a wrapper disappeared with it.

**Parity: bit-exact.** 700 monitor records compared field for field including the
`pairs`-list order, 0 mismatches. Coverage had to be fixed first: the parity set was three
oa1 rollouts, and oa1 has 0 collisions, so it would have compared 700 records without ever
entering the restructured branch. With oa2/oa3/oa5 included the comparison covers 80
records with an obstacle contact, 2507 contact pairs and 94 records with a carried object.

## Step 3: parity

gate10, N=1, B=1, D=0, `PARITY=1`:

| check | required | measured |
|---|---|---|
| collisions | 10/10 | **10/10** |
| mean min clearance | 0.00 cm | **0.000 cm** |
| completions | 3/10 on starts 4, 6, 9 | **3/10 on [4, 6, 9]** |

Action parity against chunks recorded from the unmodified `curve_gate.py`: **9 of 10
rollouts bit-exact, max |action diff| 0.000e+00 over 312 of 342 chunks**, including all
four 300-step rollouts. The exception is oa1 init 6: bit-exact for 3 chunks, then departs
at control step 24 (3.9e-03) and runs 239 steps against the reference's ~208; both collide
and both complete. So max |action diff| is 0 where trajectories stay aligned and is **not**
0 across the whole set. oa1 init 6 is independently marginal — it is the other start that
differed between M0 and M2 — and the policy now runs in a different process from the env,
so kernel selection and EGL rendering need not match the in-process reference bit for bit.

## Step 3: worker scaling

As specified, B=N and D=10 ms, on gate10:

| N | B | rollouts/h | GPU 1 sm | gate collisions | completions |
|---|---|---|---|---|---|
| 1 | 1 | 114.5 | 17.6% | 10/10 | 3 on [4,6,9] |
| 2 | 2 | 209.1 | 31.7% | 10/10 | 3 on [4,6,9] |
| 4 | 4 | 292.3 | 44.0% | 10/10 | **1 on [9]** |
| 8 | 8 | **OOM** | — | — | — |

`B=N=8` OOMs the **server**: eight EGL render contexts plus B=8 activations do not fit
beside ~20 GB of weights on a 23.68 GB card. And B>1 costs the gate's completions — at
N=4 only 23 of 339 forwards actually batched (6.8%), and that was enough to flip inits 4
and 6 to the 300-step cap. At N=1 and N=2 every forward was B=1 and the results matched
parity exactly.

Since batching is worth at most 1.27x, strict B=1 with more workers is better on both
axes:

| N | B | manifest | rollouts/h | GPU 1 sm | result |
|---|---|---|---|---|---|
| 4 | 1 | gate10 | 326.9 | 46.5% | 10/10, 3 completions |
| 8 | 1 | gate10 | 342.4 | 50.6% | 10/10, 3 completions |
| 16 | 1 | gate10 | 422.8 | 59.9% | 10/10, 3 completions |
| 8 | 1 | ctrl60 | **539.3 / 542.1** | 85.0% | 29 coll, 25 compl |
| 16 | 1 | ctrl60 | 544.7 | 80.8% | 29 coll, 25 compl |

gate10 is too small to measure scaling past N≈4 — 10 rollouts over 16 workers measures
latency, not throughput — so the knee is read off ctrl60: **N=16 beats N=8 by 1% and the
GPU is already at 85%, so the knee is N=8.** DESIGN.md predicted the knee near N=4 and
600-730 rollouts/h; the knee came later and the ceiling lower.

## Step 3: batch delay sweep, and why batching is not the default

ctrl60, N=8, B=4. Acceptance is 29 collisions and completions in {25, 26}.

| D (ms) | rollouts/h | collisions | completions | Tier B | diagonal | collision flips | max Δclearance |
|---|---|---|---|---|---|---|---|
| 1 | 623.5 | 30 | 24 | **FAIL** | 50/60 | 5 | 5.33 cm |
| 5 | 621.7 | 30 | 24 | **FAIL** | 54/60 | 3 | 5.23 cm |
| 10 | 623.5 | 29 | 26 | passes aggregate | 53/60 | **4** | 5.48 cm |
| 20 | 618.1 | 29 | 24 | **FAIL** | 53/60 | 4 | 5.37 cm |

D barely moves throughput (618-624 rollouts/h) and does not fix correctness. **D = 10 ms
passes the aggregate acceptance by cancellation, not by being right**: it still flips four
collision verdicts (oa3 inits 0, 1, 6 and oa5 init 6) and they happen to net to zero. That
is precisely why the per-start transition matrix was asked for — the aggregate count can
pass while individual starts are wrong.

So B=4 buys **+15%** throughput for 3-5 wrong collision verdicts out of 60 and clearance
errors up to 5.5 cm. **Default: B=1.**

## Step 3: Tier B at the accepted configuration

ctrl60, N=8, B=1, D=0 — and N=16 for comparison:

| | collisions | completions | transition matrix | collision flips | completion flips | max Δclearance |
|---|---|---|---|---|---|---|
| M2 reference | 29 | 25 | — | — | — | — |
| **N=8, B=1** | **29** | **25** | 60/60 diagonal | **0** | **0** | 0.1437 cm |
| N=16, B=1 | 29 | 25 | 60/60 diagonal | **0** | **0** | **0.0000 cm** |

The transition matrix is perfectly diagonal at both: (0,0)→(0,0) 11, (0,1)→(0,1) 20,
(1,0)→(1,0) 24, (1,1)→(1,1) 5. At N=16 the 60-start run is **bit-identical** to M2's
single-process run, including oa6_L1 init 6 — the start that flipped between M0 and M2.

## Per-request overhead, and what M3b would be compared against

Worker-side round trip, decomposed with the server's own reported timings:

| | N=8, B=1 | N=8, B=4, D=10 |
|---|---|---|
| round trip (worker side) | 1014.7 ms (p95 1198.3) | 794.7 ms (p95 983.5) |
| server queue wait | 803.8 ms | 289.6 ms |
| GPU forward | 184.7 ms | 473.2 ms |
| **transport + server preprocessing** | **26.2 ms (2.6%)** | **32.0 ms (4.0%)** |

The queue wait dominates because the GPU is saturated: eight workers share one card, so a
request waits behind others' forwards. That is the intended state — it is what 85% GPU
utilisation means — and it does not affect results, because each worker blocks and the
environment does not advance while it waits.

**M3a's serving overhead is 26 ms per request**, of which ~22 ms is the server's CPU
preprocessing (crop, normalise, tokenise) and the remainder the ZeroMQ round trip for a
393 KB request. That is the number a Triton comparison would be made against.

## Step 4: M3b could not be run here

Triton for Linux x86_64 ships only as a container image, and this account cannot run
containers: no `tritonserver` binary; `/usr/bin/docker` exists but `/var/run/docker.sock`
is `root:docker` mode 660 and this account's only group is `vchandr2`; no podman,
singularity, apptainer, nerdctl or enroot; PyPI's `tritonserver` is a 1.5 KB
`0.0.1.dev0` placeholder. The model repository entry and Python backend are written and
complete in `fastvla/server/triton/` — same `BatchedPolicy`, `max_batch_size 4`,
`dynamic_batching` with `max_queue_delay_microseconds 10000` — so the comparison is one
`docker run` away once container access exists, which is a prerequisite of M5 anyway.

Recorded now so it is a prediction and not a later rationalisation: Triton's dynamic
batcher is a better implementation of a mechanism these measurements show is not the
bottleneck (1.27x at best, flat past B=4, and it breaks per-start verdicts). Its value to
this project is the packaging, metrics and model-repository story in M5.

## Where M3 leaves the harness

| | M0 | M2 | M3 (N=8, B=1) |
|---|---|---|---|
| rollouts/hour | 71.2 | 166.5 | **542.1** |
| vs previous | — | 2.34x | **3.26x** |
| vs M0 | — | 2.34x | **7.6x** |
| GPU 1 utilisation | 9.9% | 26.1% | **85.0%** |
| collisions / completions | 29 / 26 | 29 / 25 | 29 / 25 |

The GPU is now the bottleneck at 85% utilisation and 6 forwards/s, so the next gain has to
come from the forward itself — which is M4, and which the M2 handoff already flagged as
having crossed the audit's 30%-of-step threshold.

---

# M3b: Triton, measured

Unblocked by `fastvla/server/CONTAINERS.md`: unprivileged Apptainer runs the NVIDIA Triton
23.07 image, and the GPU gate (section 8 there) passes. Started with
`bench/triton_serve.sh`, stopped with `bench/triton_stop.sh`.

## What it took to get the Python backend running

Recorded because none of it is in the Triton docs for this situation, and each step was a
separate failed load:

| failure | cause | fix |
|---|---|---|
| `specifies invalid or unsupported gpu id 1 ... GPUs ... are: 0` | `CUDA_VISIBLE_DEVICES=1` already selects the card and renumbers it, so inside the container the only CUDA device is 0 | `gpus: [ 0 ]` in config.pbtxt; the instance's real card is logged with its UUID at load |
| `ModuleNotFoundError: No module named 'libero'` | libero, openvla_oft and robosuite are **editable** installs whose `__editable__*.pth` files install import finders, and `.pth` files run only for directories registered through `site`, never for a directory merely on `PYTHONPATH` | `site.addsitedir(SITE_PACKAGES)` at the top of `model.py` |
| `ImportError: libX11.so.6` | the Triton image has no X or GL libraries; robosuite pulls them in through libero | a curated `bench/triton_libs/` of **graphics libraries only** |
| `ImportError: Error importing numpy: you should not try to import numpy from its source directory` | putting the env's whole `lib/` on `LD_LIBRARY_PATH` shadows the container's `libtinfo` and `libstdc++`, and numpy's C extension then fails to load. The giveaway was `bash` itself warning about the env's `libtinfo.so.6` | expose only `bench/triton_libs`, and append rather than prepend |
| `ImportError: libXext.so.6` after linking the env's X libs | the env has no libXext, and symlinks into the host's `/usr/lib/x86_64-linux-gnu` **dangle inside the container**, where that path is the container's own | copy the host's X libraries into `bench/triton_libs` rather than symlink them |
| `Incorrect path_or_model_id: ''` | `triton_serve.sh` passed `CKPT=""` and `os.environ.get("CKPT", default)` returns `""`, not the default | set CKPT in the launcher; use `os.environ.get("CKPT") or default` in the backend |
| `unexpected shape for input 'wrist_image'` | config.pbtxt said `256x256x3`, taken from this document's own (wrong) claim about the payload. The worker sends `prepare_observation`'s output, already resized to 224x224 | `dims: [-1, -1, 3]`, and the payload figure corrected above to 301 KB |

The model instance logs its device at load, which is the same evidence the pre-flight
produces:

    [triton] model instance on cuda:0 device_count=1 pci=0000:C1:00.0
             uuid=GPU-040542a1-6ad7-1ddd-f7ca-595acfd703a7

## Parity through Triton (gate10, one worker)

| check | required | measured |
|---|---|---|
| collisions | 10/10 | **10/10** |
| completions | 3/10 on starts 4, 6, 9 | **3/10 on [4, 6, 9]** |
| mean min clearance | 0.00 cm | **0.000 cm** |

Nine of ten rollouts have step counts identical to the ZeroMQ parity run; the tenth is
oa1 init 6 (239 against 204), the marginal start that has differed in every comparison
since M0.

## 60 starts at N=8, Triton against ZeroMQ

Acceptance (`bench/accept.py`, the M4 rule) on the Triton run: **29 collisions, 25
completions, 0 collision flips, 0 completion flips, 60/60 diagonal, max per-start
clearance difference 0.0072 cm. PASS.**

| | ZeroMQ (M4_T1) | Triton 23.07 |
|---|---|---|
| **rollouts/hour** | **544.8** | **511.2** (−6.2%) |
| wall, 60 starts | 396.5 s | 422.6 s |
| collisions / completions | 29 / 25 | 29 / 25 |
| acceptance | PASS | PASS |

### Where the 6.2% goes

The per-request split needs care: Triton reports queue and compute time only in aggregate,
so the first attempt differenced nothing and read statistics that still included the
parity run — which diluted the queue average and produced a nonsense overhead of 132 ms.
Re-measured against a before/after snapshot of a freshly started server (1535 requests):

| per request | ZeroMQ | Triton |
|---|---|---|
| worker round trip | 1008.2 ms | 751.3 ms |
| server queue | 798.1 ms | 547.7 ms |
| GPU forward / `compute_infer` | 184.2 ms | **200.3 ms** |
| `compute_input` + `compute_output` | — | 0.21 ms |
| **transport (residual)** | **25.9 ms** | **3.1 ms** |

Two findings, and they point opposite ways:

1. **Triton's transport is much cheaper: 3.1 ms against 25.9 ms.** Its gRPC path is C++ on
   both ends; the ZeroMQ client encodes and decodes a JSON header per request in Python and
   the server hands replies through a second queue. That is a real result against my own
   implementation.
2. **Triton's service time is higher: 200.3 ms against 184.2 ms**, and that is what sets
   throughput. The Python backend does `prepare_request` — the ~21 ms of crop, normalise
   and tokenise — *inside* `execute`, serialised with the GPU forward. The ZeroMQ server
   deliberately splits that across two threads, so preprocessing overlaps the forward and
   never reaches the critical path.

Throughput follows service time, not transport, because a worker blocks on its own request
and its environment does not advance while it waits: 1000/184.2 = 5.43 forwards/s against
1000/200.3 = 4.99, a predicted 8.1% against the 6.2% measured. The queue numbers are not
comparable between the two columns (the clean Triton run had 6 live workers, see below), but
the service and transport figures do not depend on concurrency.

The fix for Triton would be to get preprocessing off `execute` — a two-model ensemble, or
preprocessing in the client — not to change the model. It was not pursued: it would make
the comparison a comparison of two different request protocols rather than of two servers.

### A robustness caveat

In one Triton run, **2 of 8 workers died with SIGSEGV at startup**, before producing any
rollout (45/60 rollouts completed; the other six exited cleanly). The earlier Triton run
had 8 of 8 succeed, so it is intermittent, and it appears only on the Triton path — the
likely interaction is gRPC's threads with EGL context creation in a process that also
loads MuJoCo. The gRPC client also raises a harmless `AttributeError` in `__del__` at
interpreter shutdown on every worker. Neither affects results, because an incomplete run
fails the acceptance check on row count, but it is a reason the ZeroMQ path is the one to
benchmark with.

## Which is which

- **Benchmark default: the ZeroMQ server.** It is faster (544.8 against 511.2
  rollouts/hour), it has produced 8-of-8 complete worker runs every time, and its
  two-thread design keeps preprocessing off the critical path.
- **Deployment option: Triton.** Not because it is faster here — it is not — but because it
  brings the model repository, versioning, a metrics endpoint and a standard client, and
  because its transport is genuinely cheaper. On a box where the forward is not the
  bottleneck, or with preprocessing moved off `execute`, the ranking would likely reverse.

Both are kept. `POLICY_TRANSPORT=zmq|triton` selects one, the workers and manifest are
identical either way, and the acceptance rule is applied to both.
