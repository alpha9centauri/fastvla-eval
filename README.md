# FastVLA-Eval

Closed-loop evaluation of vision-language-action policies is slow, and the slowness is
usually not where people think it is. FastVLA-Eval runs the LIBERO-Safety benchmark suite
against an OpenVLA-OFT policy with many simulator workers feeding one batched policy
server, a C++ clearance labeler in place of the Python one, and a per-start correctness
gate that every change has to pass. It began as an audit: the first measurement showed the
GPU idle 90% of the time and a geometry routine, not the 7B model, eating the wall clock.

**71.2 → 544.8 rollouts/hour on one RTX A5000 — 7.6× — with zero collision-verdict changes
across the 60 fixed starts.**

<p align="center">
  <img src="docs/media/rollout.gif" width="480"
       alt="A simulated Franka arm reaching past an obstacle to pick up an object">
</p>

<sub>One rollout from this harness (`oa1`, init 0): completes the task, never contacts the
obstacle, closest approach 5.84 cm. 226 control steps shown at 1.6× speed.
Regenerate with `VIDEO_KEYS=oa1__i00__base bench/m3a_run.sh demo6 6 1 0`.</sub>

## Benchmark

<!-- BENCH TABLE START -->
Every row measured on **RTX A5000 (GPU 1 of 2)**, driver **530.30.02**, torch **2.2.0+cu121** (constant across rows -- one box, one environment). The commit column is the commit that last touched that row's artefact, i.e. when the number was last measured. Reproduce with `make bench`.

| metric | before | after | how measured | commit |
|---|---|---|---|---|
| rollouts/hour, 60-start eval, one RTX A5000 | 71.2 (M0, single process) | **544.8** | `bench/bench_all.sh` → `bench/out/m4_t1_summary.json` | `28deac3` |
| ms per control step, per worker (8 workers sharing one GPU) | 190.3 (M0, one process) | 197.7 | wall x N_workers / total steps; higher than M0 because 8 workers contend for one GPU | `28deac3` |
| GPU 1 SM utilisation during the eval | 9.9% (M0) | 85.4% | `nvidia-smi dmon -s um`, 1 Hz → `bench/parse_dmon.py` | `28deac3` |
| clearance labeler, ms per call (fires twice per control step) | 63.64 | **0.578** | `bench/call_run.sh` → `bench/out/m2_call_*.json` | `348ba38` |
| clearance labeler, µs per exact hull pair | 634 | 5.7 | same, exact pairs counted per call | `348ba38` |
| MuJoCo contact scan, ms per control step | 8.19 | **2.25** | `audit/t2_run.sh` CONTACTS=0 vs 1, identical instrumentation | `e902380` |
| policy forward, ms per call, batch 1 (bf16, 2 cameras + proprio, 8-step chunk) | — | 167.97 | `bench/fwd_run.sh time baseline` | `28deac3` |
| forward split: 7B language model / vision backbone | — | 130.2 / 33.9 ms | same run, CUDA events per submodule | `28deac3` |
| per request: queue wait / GPU forward / transport+prep, at 8 workers | — | 798.1 / 184.2 / 25.9 ms | worker round trip minus server-reported parts | `28deac3` |
| serving layer: our ZeroMQ server vs Triton 23.07, same model | 544.8 roll/h | 511.2 roll/h | 6% slower: service time +16 ms because `prepare_request` runs inside `execute`; its transport is cheaper, 3 vs 26 ms (`fastvla/server/DESIGN.md`) | `3d8cc12` |
| same eval inside Apptainer (unprivileged, --nv) | 544.8 roll/h (host) | **547.5 roll/h** | `bench/incontainer_run.sh` | `02aa037` |
| cold start, server launch → first request served | — | 15.0 s | same run, image pull excluded | `02aa037` |
| correctness: collisions / completions on the 60 fixed starts | 29 / 26 (M0 reference) | **29 / 25** | `bench/accept.py` (requires 29, and 25 or 26) | `28deac3` |
| correctness: per-start collision-verdict flips vs the reference | — | **0** | same; the run-to-run floor is 0 collision flips (`fastvla/monitor/EVAL.md` section 7) | `28deac3` |
| correctness: largest per-start min-clearance difference vs the reference | — | 0.1530 cm | same | `28deac3` |
| labeler stages (python): union-hull GJK reject / exact pair GJK | — | 22.5 / 40.2 ms | `bench/stages_run.sh` | `b9a22b7` |
<!-- BENCH TABLE END -->
Every row is read from a JSON artefact under `bench/out/` that a bench script wrote;
`make bench-check` fails if this table and the artefacts disagree, so the README cannot
drift from what was measured.

## The runtime collision monitor

The second result. The policy's own activations already know how close the arm is to the
obstacle, so a linear probe on them is a collision monitor that costs almost nothing and
needs no extra sensor. It is built into the server behind `MONITOR=off|log|stop`, off by
default. Method and full results: [`fastvla/monitor/EVAL.md`](fastvla/monitor/EVAL.md).

| | unshifted (60 starts) | intrusion-5 (48 starts, never trained on) |
|---|---|---|
| Spearman, predicted vs true clearance | **+0.868** | +0.480 |
| AUROC, contact within 8 control steps | **0.918** | 0.743 |
| cost per forward | **0.067 ms**, 0.037% of a 183 ms forward | same |
| throughput with `MONITOR=log` | **546.2 rollouts/hour** vs the 544.8 baseline | — |

`MONITOR=log` leaves the acceptance gate untouched — 29 collisions, 25 completions, zero
collision flips, 60/60 diagonal — which it must, being a dot product taken after the
forward has returned, outside the action path.

The operating point is a threshold plus a 2-forward debounce, chosen at the knee of the
recall / false-alarm curve on the **training** scenes only:

<p align="center">
  <img src="fastvla/monitor/operating_curve.png" width="760"
       alt="Recall against false alarms for the collision monitor, two test sets">
</p>

**`MONITOR=stop` is a brake, not an avoidance.** It zeroes the pose delta and holds the
gripper for the rest of the rollout. It has no notion of which direction is safe, it never
resumes, and it cannot steer around anything:

| `MONITOR=stop` vs `MONITOR=off`, per start | unshifted (60) | intrusion-5 (48) |
|---|---|---|
| collisions | 29 → **11** (18 prevented) | 40 → **28** (12 prevented) |
| starts made worse (newly colliding) | **0** | **0** |
| completions | 25 → **12** (13 lost) | 4 → **0** (4 lost) |
| rollouts braked | 46 of 60 | 43 of 48 |
| median first hold | control step 92 of 300 | control step 88 |
| median lead, first alarm → contact | **74 control steps** | 78 |

The 18 collisions prevented and the 13 completions lost are the same trade, and quoting
either alone would misrepresent it. An earlier pass used a 90%-recall threshold with no
debounce: it alarmed on every clean rollout, braked all 108 rollouts at a median of
control step 8, and drove collisions to 1/60 and completions to 1/60 — the arm stopped
moving almost before it started. Both runs are kept in EVAL.md section 9, because the
contrast is the argument for the operating point above.

## How it works

```
  manifest: 60 (task, init, arm) triples, fixed
        |
        v
  +-----------------------------------------------+
  |  runner: spawns N workers, one shared server   |
  +-----------------------------------------------+
        |                                     |
        v                                     v
  8 x simulator worker                  policy server (1 GPU)
  +---------------------------+         +------------------------------+
  | MuJoCo env (one process)  |  ZeroMQ | receiver thread: preprocess  |
  | observation --------------+-------->| (crop, normalise, tokenise)  |
  |                           |  ipc:// |            |                 |
  | C++ GJK labeler:          |         |            v                 |
  |   clearance + contact,    |         | forward thread: one GPU      |
  |   0.578 ms per call       |         | forward per batch            |
  |                           |         |            |                 |
  |                           |         |   +--------+---------+       |
  | 8-step action chunk <-----+---------+---| monitor: probe on |      |
  +---------------------------+         |   | the activations   |      |
        |                               |   +-------------------+      |
        v                               +------------------------------+
  per-rollout npz: eef, clearance,
  contact, actions, activations
        |
        v
  bench/accept.py: 29 collisions, 25-26 completions,
  ZERO per-start collision flips -- or the change is rejected
```

The monitor sits between the forward and the reply, so `MONITOR=stop` brakes the very
chunk whose activation raised the alarm. The activation leaves the server only when a
collection run asks for it, so `log` costs no bytes on the wire.

<p align="center">
  <img src="docs/media/harness.gif" width="479"
       alt="Terminal recording: GPU pre-flight, server start, eight workers, rollouts completing">
</p>

<sub>A real run of the 10-start gate set on GPU 1, replayed at 12.5× — the GPU pre-flight
that refuses to start on the wrong card, the server coming up, eight workers, and the
throughput counter as rollouts land. These ten starts are all intrusion-5 scenes, which
collide and run to the 300-step cap, so they are slower than the 60-start figure above.
The transcript it was rendered from is `docs/media/harness_session.txt`; the renderer is
`bench/render_terminal_gif.py`.</sub>

## Correctness

Speed is worthless here if the verdicts move, so every change is gated per start, not on
aggregates.

**Parity mode.** `PARITY=1` pins N=1, B=1, D=0 and asserts it. In that mode
`BatchedPolicy.forward` on a single request is bit-exact against the reference
`get_action` — max |difference| 0.000e+00 over 16 real re-plan observations — and the
worker loop reproduces the reference driver's four invariants.

**The 10-start gate.** `gate10`, parity mode, against the reference driver: 10/10
collisions, 3/10 completions on starts 4, 6 and 9, mean min clearance 0.000 cm, and action
parity of 0.000e+00 over 312 of 342 chunks.

**The 60-start acceptance.** `bench/accept.py` requires 29 collisions, completions in
{25, 26}, and **zero per-start collision-verdict flips**. The transition matrix is
diagonal at all 60: (no collision, no completion) 11, (no collision, completion) 20,
(collision, no completion) 24, (collision, completion) 5.

The aggregate counts alone are not enough, which is the whole reason the per-start matrix
exists — and both rejected speed-ups below prove it:

| rejected | what it bought | why it was rejected |
|---|---|---|
| dynamic batching, B=4 | 618–624 rollouts/hour, +13% | **3 to 5 flipped collision verdicts** at every delay tried. At D=10 ms it passes the aggregate count (29/26) by cancellation while four individual starts are wrong |
| `torch.compile` | forward 168.0 → 156.7 ms, −6.7% in the microbenchmark | **3 flipped verdicts and −15% end to end** (459.0 rollouts/hour): it won the microbenchmark and lost the real run |

**The run-to-run floor.** Two runs of the same manifest in the same configuration are not
always trace-identical: two capture-off runs differ on 4 of 60 traces. It is not the
instrumentation — in all 8 divergent traces the first difference falls inside the first
control step of a forward, so the simulator is deterministic and the policy forward is not
bit-reproducible run to run at `TORCH_DETERMINISTIC=0`, the configuration every number
here was measured in. The floor is **zero collision flips on 60 starts; completions within
0 to 1**, measured in
[`fastvla/monitor/EVAL.md` section 7](fastvla/monitor/EVAL.md). Every claim above clears
it; "identical" anywhere in this repo means that, not bitwise equality of trajectories.

## Install

```bash
git clone git@github.com:alpha9centauri/fastvla-eval.git && cd fastvla-eval
python -m pip install -e .     # the fastvla package; declares no dependencies, on purpose
make labeler                   # builds the C++ GJK labeler in place (g++ -O3 -fopenmp)
```

`make test` then runs the CPU suite — 50 tests, no GPU and no checkpoint — and is the
check that the install worked.

The package deliberately declares **no runtime dependencies**. Rollouts need torch,
MuJoCo, robosuite, LIBERO-Safety, OpenVLA-OFT and a checkpoint, none of which are vendored
here; pinning them in `pyproject.toml` would invite pip to resolve a second, different
stack underneath a measurement. Install into the environment that already has them —
`audit/AUDIT.md` records the exact one every number was measured in — or use the image in
`deploy/`.

## Reproduce the numbers

```bash
make bench                          # every bench script, writes bench/out/*.json (~25 min)
make bench-quick                    # the end-to-end rows only (~8 min)
make bench-table                    # regenerates the table above from those artefacts
make bench-check                    # fails if the table and the artefacts disagree
make accept RUN=<run_id>            # the 60-start gate on a run under bench/runs/
```

`make bench` needs GPU 1 and the checkpoint. `make test` and `make lint` need neither.

## What is measured and what is not

Measured on this box, with the artefact named in the table: everything above.

Not measured, and not claimed:

- **The Kubernetes manifests in `deploy/k8s/` have never been applied.** There is no
  cluster here. They are written from the compose file and are untested.
- **The GPU image is not built in CI.** The runner has no GPU and no checkpoint; CI builds
  and tests the CPU image only. The GPU image is built by hand on this box.
- **Triton is 6.2% slower than the ZeroMQ server** (511.2 against 544.8 rollouts/hour) and
  is kept as a comparison, not a recommendation. Its transport is much cheaper — 3.1 ms
  against 25.9 ms — but its service time is 16 ms higher, because the Python backend runs
  `prepare_request` inside `execute` where it cannot overlap the GPU forward. The ZeroMQ
  server overlaps preprocessing with the forward on a separate thread.
- **One GPU, one box, one policy.** Nothing here has been run on a second machine, a
  second GPU model, or a second policy family.

## Limitations

- The monitor's probe is fitted to this policy and these scenes. On intrusion-5, which it
  never trained on, Spearman falls from +0.868 to +0.480 and it catches 12 of 40
  collisions rather than 18 of 29.
- `MONITOR=stop` is a brake. It prevents collisions by stopping, and it costs completions
  at every operating point measured.
- Batch sizes above 1 change action values (bf16 reduction order) enough to flip verdicts,
  so the fast path is B=1 and the throughput comes from parallel simulators, not batching.
- 60 starts is a small set. Verdict-level differences of one or two starts are inside the
  run-to-run floor and should not be read as signal.
- The acceptance reference is a run, not ground truth. It fixes behaviour against a known
  point; it does not establish that the point is correct.

## Acknowledgements

- [LIBERO-Safety](https://github.com/LIBERO-SAFETY/LIBERO-Safety) — the benchmark suite,
  and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) upstream of it.
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft) — the policy and the fine-tuning
  recipe this harness evaluates.
- [openpi](https://github.com/Physical-Intelligence/openpi) — reference for the policy
  serving patterns.
- **Heng Zhang** and the **Safe AI Lab at Carnegie Mellon University**, for the research
  this harness was built to serve.

## Licence

MIT, see [LICENSE](LICENSE). Third-party components keep their own licences — LIBERO and
robosuite MIT, LIBERO-Safety and openpi Apache-2.0, OpenVLA-OFT MIT — and are listed with
their sources in [`docs/PROVENANCE.md`](docs/PROVENANCE.md), which also records how every
external fact, URL and version pin in this repository was verified.
