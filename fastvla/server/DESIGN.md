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
- **Images go as raw uint8** (2 x 256 x 256 x 3 = 393 KB), not as prepared tensors. The
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
