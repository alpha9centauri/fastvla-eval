# How we know the answers did not change

Making an evaluation faster is worthless if it changes what the evaluation says. This page
is how that is checked.

## The vocabulary

A **rollout** is one trial: one task, one starting arrangement of the objects, up to 300
control steps. A **collision verdict** is the yes/no answer for that trial — did the robot
touch the obstacle. A **completion** is whether it finished the task. The **60-start set**
is sixty fixed trials, six tasks by ten starting arrangements, and it is the standard test.

The point of the checks below is that the *per-trial* answers stay the same. Matching
totals is not enough, and section "Two speed-ups that were rejected" shows why.

## Parity mode

`PARITY=1` pins the harness to one worker, batch size one, no batching delay, and asserts
all three. In that mode the batched policy wrapper is bit-exact against the reference
implementation it replaced: maximum absolute difference **0.000e+00** over 16 real
re-plan observations. That comparison runs both paths in one process. It says the wrapper
adds no arithmetic; it does not say a forward repeats across runs, which it does not — see
"The run-to-run floor" below.

The worker's rollout loop also reproduces the reference driver's four invariants: gripper
post-processing before the environment step, ten dummy warm-up steps with the monitor
indexed from the end of them, completion read from the environment's own done flag, and
the scene-setup order.

## The 10-start gate

A fast check against the original driver, in parity mode, on ten trials:

| check | required | measured |
|---|---|---|
| collisions | 10/10 | **10/10** |
| completions | 3/10, on starts 4, 6 and 9 | **3/10 on [4, 6, 9]** |
| mean minimum clearance | 0.00 cm | **0.000 cm** |

Action parity against chunks recorded from the unmodified reference: **no difference at
all over 312 of 342 chunks**, including all four 300-step trials. The exception is one
start that departs at control step 24 by 3.9e-03 and runs longer than the reference; both
runs collide and both complete. That start is independently marginal — it is the one that
also differed between the original harness and the C++ labeler port.

## The 60-start acceptance gate

`bench/accept.py` is the gate every change has to pass. It requires:

- **29 collisions**, exactly;
- **completions in {25, 26}**;
- **zero per-start collision-verdict flips** against the reference run.

The transition matrix is diagonal at all 60 trials: (no collision, no completion) 11,
(no collision, completion) 20, (collision, no completion) 24, (collision, completion) 5.

```bash
RUN_ID=my_change bash bench/m3a_run.sh ctrl60 8 1 0
make accept RUN=my_change
```

## Two speed-ups that were rejected

Both made the harness measurably faster. Both changed the answers.

| rejected | what it bought | why it was rejected |
|---|---|---|
| dynamic batching, batch size 4 | 618–624 rollouts/hour, about +13% | **3 to 5 flipped collision verdicts** at every queue delay tried |
| `torch.compile` | forward 168.0 → 156.7 ms, −6.7% in a microbenchmark | **3 flipped verdicts and −15% end to end** (459.0 rollouts/hour) |

Batching is the instructive one. At a 10 ms delay it reports 29 collisions and 26
completions — it *passes* the aggregate acceptance. It also flips four individual verdicts,
and they happen to net to zero. A reviewer checking totals would have shipped a
configuration that got four of sixty trials wrong. That is why the gate is per trial.

`torch.compile` is the other lesson: it won the microbenchmark, which timed the forward
alone after warm-up, and lost the real run. Eight workers send two different prompt
lengths, the compiled graph re-specialises, and recompilation costs more than the
per-call saving.

Both are written up with full numbers in `fastvla/server/DESIGN.md` and
`fastvla/server/FORWARD.md`.

## The run-to-run floor

Two runs of the same trials in the same configuration are not always identical. Measured:

| pair | instrumentation | traces differing | max clearance diff | collision flips | completion flips |
|---|---|---|---|---|---|
| two 60-start runs | both capture off | 4 of 60 | 5.3e-3 m | 0 | 0 |
| 60-start, capture on vs off | differs | 2 of 60 | 5.3e-3 m | 0 | 0 |
| 60-start vs the 16-worker reference | differs | 0 of 60 | 0 | 0 | 0 |
| 48-start intrusion set | differs | 4 of 48 | 7.2e-2 m | 0 | **1** |

It is not the instrumentation: two capture-off runs differ from each other as much as any
other pair. In all 8 divergent traces the first difference falls inside the first control
step of a forward — never mid-chunk, where a simulator discrepancy would show. The
simulator is deterministic. The policy forward is not bit-reproducible run to run at
`TORCH_DETERMINISTIC=0`, which is the configuration every number in this repository was
measured in. Determinism mode was measured at 526.2 rollouts/hour, −2.9%, and not adopted.

So the floor for any per-trial comparison here is **zero collision flips on 60 trials, and
completions within 0 to 1**. Every claim in this repository clears it. "Identical" in these
documents means that, not bitwise equality of trajectories. The full measurement is in
`fastvla/monitor/EVAL.md` section 7.

## Triton, measured and not adopted

NVIDIA's Triton Inference Server was benchmarked against the ZeroMQ server in this
repository, serving the same model.

| | ZeroMQ server | Triton 23.07 |
|---|---|---|
| rollouts/hour | **544.8** | 511.2 (−6.2%) |
| worker round trip | 1008.2 ms | 751.3 ms |
| GPU forward / `compute_infer` | 184.2 ms | **200.3 ms** |
| transport (residual) | **25.9 ms** | **3.1 ms** |

Triton's transport is much cheaper — 3.1 ms against 25.9 ms, a C++ gRPC path against a
Python ZeroMQ one. Its service time is 16 ms higher, and that is what sets throughput: the
Python backend runs the ~21 ms of image preprocessing inside its `execute` call, where it
cannot overlap the GPU forward. The ZeroMQ server runs preprocessing on a separate thread,
so it overlaps. Keeping Triton would mean giving up more in service time than it wins in
transport.

## What is measured, and what is not

Measured on one box, with the artefact named in `docs/BENCHMARK.md`: every number in this
repository.

Not measured, and not claimed:

- **The Kubernetes manifests in `deploy/k8s/` have never been applied.** There is no
  cluster here. They are written from the compose file and are untested.
- **The GPU image is not built in CI.** The runner has no GPU and no checkpoint. CI builds
  and tests the CPU image only; the GPU image is built by hand.
- **One GPU, one box, one policy.** Nothing here has run on a second machine, a second GPU
  model, or a second policy family.
- **The acceptance reference is a run, not ground truth.** It fixes behaviour against a
  known point. It does not establish that the point is correct.
