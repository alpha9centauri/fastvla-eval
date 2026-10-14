# FastVLA-Eval: a fast, correct closed-loop evaluation harness for VLA policies

Project plan and design. Written for a coding agent that will build it on the shared GPU box
after the current research wraps up. Owner: Varun Chandra (vchandr2). Author on every commit
and file is Varun; no AI attribution anywhere; nothing is pushed to any remote without Varun's
explicit OK.

Read this whole file before touching anything. Then do the audit in section 4 before writing
code. Every number in this plan that is not marked MEASURED is an estimate and must be replaced
by a measurement.

Revision 2, 13 Oct 2026. The research closed on 13 Oct (data chapter verdict in
`behavior/displacement/datagen/DATA_CHAPTER_VERDICT.txt`). This plan now starts from a cleaned
box: one git repo for the research code, a `PROJECT_HANDOFF.md` written by the previous agent
with box specifics (paths, conda envs, the reference commands and numbers), and about 240 GB
of old checkpoints removed. Changes in this revision: section 7a (motion planner milestone),
section 7b (what makes it resume-worthy beyond the code), section 9 (git and GitHub rules now
that a remote exists), section 13 (what Varun still has to specify). Varun will add build
context before the first prompt is sent; see section 13.

---

## 1. Aim

Turn the evaluation and training infrastructure built during the SafeSteer research (VLA
safety interpretability, LIBERO-Safety, OpenVLA-OFT and Pi0.5) into one finished, open-sourceable
systems project that:

1. Makes closed-loop VLA evaluation on LIBERO-Safety roughly an order of magnitude faster
   (target 5-10x rollouts per hour; MEASURE the baseline first) without changing results.
2. Proves, with numbers, the following skills for ML systems / ML infra / DevTech roles:
   performance profiling (Nsight Systems, Nsight Compute, torch.profiler), C++ with pybind11,
   batched GPU inference serving (NVIDIA Triton), GPU inference optimisation (bf16,
   torch.compile, CUDA graphs), multiprocess simulation infrastructure, correctness testing under
   nondeterminism, containerised deployment (Docker, a Kubernetes manifest), and one justified
   CUDA kernel with a profile that shows where it wins and where it does not.
3. Ships the two LIBERO-Safety scoring-bug fixes and the GJK clearance labeler as part of a
   harness other labs can run.

Non-goals: distributed training (FSDP, NCCL) and production cloud scale. Do not claim them.
If a second GPU becomes available for a weekend, a DDP run is an optional appendix, nothing more.

Time budget: four to five weeks part time. Each milestone stands alone and ends with one
resume-ready number. Order is by payoff; stop after any milestone and the project is still
coherent.

---

## 2. Target roles (why these skills)

Two NVIDIA job descriptions drove the skill list.

AI DevTech Engineer: C/C++ fluency, low-level performance optimisation, parallel programming,
modern CPU and GPU architecture, performance analysis. Served by milestones 1, 2, 4, 6.

Systems Software Engineer, AI and Cloud: Triton Inference Server or vLLM, Kubernetes,
microservices, API design for inference systems, developer tutorials. Served by milestones 3, 5, 7.

Both value: measured before/after numbers, open-source contributions, public technical write-ups.

---

## 3. What exists today (starting point)

All paths are under `/data_new/vchandr2/` on gpu5 unless stated. Since the 14 Oct reorganisation
the research code lives in the git repo `vla-safety-interp/` (this work: `src/displacement/`),
checkpoints in `checkpoints/`, datasets in `data/`, the review page in `review/`; the older paths
named below (`behavior/displacement/`, `runs/`) still resolve through symlinks. `PROJECT_HANDOFF.md`
in `src/displacement/` has the current paths; it wins over any path in this section. GPU 0 is usually occupied by
another user's long-running job (never touch it); GPU 1 is ours. Disk: 453 GB free on
`/data_new` at 97% on 13 Oct before cleanup; the cleanup frees about 240 GB. Re-check with `df`. Conda envs: `libero-vla-finetune` (OpenVLA-OFT, TFDS, training) and a separate
LIBERO/openpi env for simulation and Pi0.5. Check `conda env list` and the shebangs in
`run_with_alert.py` and `behavior/displacement/*.py` for which env runs what.

### 3.1 Policies
- OpenVLA-OFT 7B, our fine-tune, checkpoint family "17500" (`runs/...17500...`): Llama-2 backbone,
  33-layer residual stream, hidden 4096, L1 regression action head reading the post-final-norm
  hidden at the action-token slice, 8-step action chunk, two camera images (agentview + wrist,
  256x256) plus proprio (8-d). Inference is one forward per 8 actions. bf16 weights.
  Training recipe: `openvla-oft/.../finetune_headinit.py`, `launch_headinit.sh` (LoRA r32 a16,
  batch 1 x grad-accum 32 on one 24 GB card). Continued-LoRA recipe used this month:
  `datagen/run_ft_liftonly.sh` (accum 8, lr 5e-5, shared normalisation stats, mixture patch
  `datagen/openvla_oft_mixture.patch`).
- Pi0.5 (openpi, JAX): PaliGemma backbone + flow-matching action expert. Served by
  `behavior/displacement/serve_pi05_noise.py` as a policy server; seeded sampling gives bit-exact
  determinism (`Policy.infer(noise=)`). Out of scope for optimisation unless time remains; keep
  it working as a second client of the harness.

### 3.2 Simulator and benchmark
- LIBERO-Safety (MuJoCo via robosuite): six tabletop pick-and-place tasks (oa1..oa5, oa6_L1),
  each with 50 saved init states; evaluation uses init 0-9 (60 fixed starts). OSC_POSE
  controller, 20 Hz, 7-d actions. BDDL task files define objects, goal predicate and the
  constraint (contact with the declared obstacle).
- Two scoring bugs we fixed in the benchmark's collision accounting (compound-'and' cost;
  cost zeroed on success). Our fixed scoring lives in the displacement driver; find it by
  grepping for `cost` handling in `behavior/displacement/rollout_openvla_disp.py` and the
  SafetyMonitor.
- Deterministic replay: `env.seed(); env.reset(); env.set_init_state(state)` reproduces rollouts
  bit-exactly on the same card (verified 241/242 and 242/242 in earlier work; one start,
  oa6_L1 init 6, is a completion coin flip on identical inputs and is excluded from counts).
  Cross-card runs can flip one marginal completion (cuDNN kernel selection), so every
  comparison runs on one card with its reference arm re-run on that card.

### 3.3 Clearance labeler (the GJK monitor)
- `SafetyMonitor` in `behavior/displacement/` (see `geom_shift.py` and the monitor module it
  imports; `HullCache` caches convex hulls per entity): per step, computes the minimum convex
  distance between robot links (plus the carried payload) and the declared obstacle, using GJK
  over per-geom convex hulls. The oa1 bottle is 244 hull pieces. Reported cost: about 543 ms per
  step with a wide cutoff (the cutoff defeats the bounding-sphere pruning); NOT independently
  measured for this plan. Known gaps: `main_table` and `microwave_1` are not in its entity
  index (bounding box returns +-inf); knocked objects are not counted.
- Validated four ways in earlier work (analytic cases, MuJoCo's own distance, contact events,
  replay). Those tests should become the parity suite.

### 3.4 Closed-loop evaluation driver
- `behavior/displacement/rollout_openvla_disp.py`: the displacement test. For each (task, init,
  arm): reset, apply the obstacle shift (`geom_shift.apply_shift`: free bodies via init state,
  static bodies via `model.body_pos`, dynamic objects via a mocap wrapper; all must be applied
  AFTER reset), then step the policy. Logs per-step clearance, collision, completion, EE path,
  optional probe readouts; writes CSV rows and videos (`oft{STEP}__{task}__i{init}.mp4`, with a
  3-tuple video spec). Runs ONE environment at a time, policy forward in the same process.
  Reference: 60 starts for one arm take on the order of 1-2 hours on GPU 1 (ESTIMATE from run
  logs; measure). The per-step cost is split between MuJoCo stepping, rendering two cameras,
  the 7B forward, and the GJK labeler; the split is NOT measured.
- `run_with_alert.py` wraps long jobs with W&B logging and e-mail alerts (RUN_LOG_NAME).
- `behavior/displacement/pullback_rollout.py`, `rollout_pi05_disp.py`, `rollout_pi05_seeded.py`:
  sibling drivers with the same scene-setup pattern.

### 3.5 Demo generator (datagen)
- `datagen/dgcore.py` and `gen_stagec.py`: scripted OSC waypoint planner with GJK-swept detours,
  certification at >= 2 cm, LeRobot v2.1 writer, render-skip (plan and certify without cameras,
  then replay the accepted actions with cameras; verified bit-identical after the F35 fix), TFDS
  conversion (`libero_safety_converter_tfds.py`). Per-episode scene setup re-applies static and
  mocap shifts after every reset. Multiprocess shards already exist here (CPU); the policy is not
  involved, so it is a model for the simulator-worker design.

### 3.6 Review page
- `behavior/displacement/review/index.html` built by `build_review.py`; served with
  `python -m http.server 8765 --bind 127.0.0.1` and viewed over `ssh -L`.

### 3.7 Existing numbers to reuse as ground truth
- 17500 on the 60 fixed starts: 29 collisions, 25 completions, 20 avoided-and-completed
  (GPU-0 anchor; on GPU 1 one completion flips, 29/26). Intrusion-5 set (48 feasible arms):
  40 collisions, 1 completion.
- These are the golden outputs for the correctness checks in section 6.

---

## 4. Audit first (milestone 0, two days)

Before any code: measure where time goes. Deliverable `AUDIT.md` with these tables.

1. End-to-end: wall-clock for one full 60-start evaluation of 17500 on GPU 1, current code.
   Rollouts per hour. Steps per second.
2. Per-step breakdown over 3 rollouts (300 steps each), using `torch.profiler` for the GPU side
   and `cProfile` plus manual timers for the CPU side: MuJoCo `env.step`, rendering (both
   cameras), observation preprocessing (resize, normalise, tokenise), policy forward (split
   vision encoder / LLM / action head if cheap to separate), GJK labeler, logging and video
   writing, Python glue. Report mean and p95 per component and GPU utilisation
   (`nvidia-smi dmon` sampled during the run).
3. Nsight Systems trace of one rollout (`nsys profile`), with a screenshot-equivalent summary:
   is the GPU idle while MuJoCo steps? How long are the gaps?
4. Labeler microbenchmark: ms per call on 1,000 saved states from an existing rollout, with the
   cutoff as used in production and with pruning enabled. Count of hull pairs per call.
5. Fine-tune data path: samples per second and GPU utilisation of the existing
   `finetune_headinit.py` at batch 1 x accum 8 for 200 steps (ESTIMATE from `ft_liftonly.log`:
   ~4.8 s/it at effective batch 8).

Decision rule after the audit: if the policy forward is under 30% of step time and the
simulator plus labeler dominate, milestone 3 (parallel workers + batched server) is the big win
and milestone 4 is small; if the forward dominates, reverse the order of 3 and 4. Write the
decision and its evidence into AUDIT.md.

Also list in AUDIT.md: every place the current driver assumes a single environment (global
state, file names, video spec), because milestone 3 has to break those assumptions safely.

---

## 5. Architecture (target state)

```
                 +------------------+        gRPC / shared memory         +---------------------+
   N x sim worker| MuJoCo + cameras |  obs (2 imgs, proprio, task id) --> |  Policy server      |
   (CPU procs)   | GJK labeler (C++)|  <-- 8-step action chunk            |  Triton, bf16 OFT   |
                 | scoring, logging |                                     |  dynamic batching   |
                 +------------------+                                     +---------------------+
                          |                                                         |
                          v                                                         v
                 results/{run}/rows.parquet, videos/, per-step npz          metrics (Prometheus)
                          |
                          v
                 analysis: transition matrix vs golden, binomial checks, review page
```

Components:
- `fastvla/sim/worker.py`: one process per environment. Owns env creation, seeded reset,
  scene shift (reuse `geom_shift` with the post-reset re-apply), stepping, rendering, the C++
  labeler, scoring with the fixed cost accounting, per-step logging to a local buffer, optional
  video. Talks to the server through a client stub. Workers are launched by a pool with a
  manifest of (task, init, arm, seed) specs and write one row per rollout.
- `fastvla/labeler/` : C++17 GJK over convex hulls, OpenMP or std::thread over hull pairs,
  bounding-sphere pruning, pybind11 module `fastgjk`. Same API as the Python `SafetyMonitor`
  (`min_clearance(state) -> (distance, actor, obstacle_piece)`), plus a batched entry point
  `min_clearance_batch(states)` used by the CUDA variant in milestone 6.
- `fastvla/server/`: Triton Inference Server with a Python backend (or PyTorch backend if the
  OFT forward can be exported cleanly) wrapping the OpenVLA-OFT forward: inputs (2 images uint8,
  proprio f32, instruction string or pre-tokenised ids), output (8x7 actions f32). Dynamic
  batching with a max queue delay (start at 5 ms), bf16, optional torch.compile and CUDA graphs
  behind flags. A `batch_size=1, no_compile` mode is the parity mode. Expose Triton metrics.
- `fastvla/client/`: thin client used by workers; also a drop-in `Policy` class so the existing
  drivers can call the server without other changes (keeps `rollout_openvla_disp.py` runnable
  as a regression reference).
- `fastvla/eval/`: manifest builder (the 60 starts, the intrusion-5 set, custom shift lists),
  pool runner, result aggregation, transition matrix against a golden CSV, binomial-error
  comparison, review-page builder (reuse `build_review.py`).
- `deploy/`: Dockerfile for the server, Dockerfile for workers, `docker-compose.yml` for the
  box, `k8s/` with a Deployment for the server and a Job for workers (documented and tested
  on a local kind cluster or minikube if available; if not, say so).
- `bench/`: scripts that produce every number in the README table from scratch.

Transport: start with gRPC (Triton's native) over localhost. If the audit shows image
serialisation is a bottleneck (2 x 256x256x3 uint8 = 393 KB per request), switch workers to
Triton's shared-memory API (CUDA shared memory or system shared memory) and report the
difference. That comparison is itself a resume number.

---

## 6. Correctness (non-negotiable, applies to every milestone)

Batching, bf16 and compile can change floating-point results, so correctness is two-tier.

Tier A, bit-exact parity mode. Server at batch size 1, no compile, same dtype as the current
driver, workers sequential. For 10 fixed (task, init) starts, the per-step actions, clearance
values, collision flags and completion flags must match the current driver's saved rollouts
exactly (max |action diff| == 0, same step count). This proves the plumbing (scene shift order,
observation preprocessing, scoring) is unchanged.

Tier B, statistical equivalence in fast mode. On the 60 fixed starts (59 after excluding
oa6_L1 init 6) and on the intrusion-5 set, collisions and completions must fall within two
binomial standard errors of the golden counts (29/25 on 60 starts: SE about 3.8 for either).
Report the transition matrix (rescued / kept / lost / coll-to-stop) against the golden rows;
expect it to be near-diagonal. Also report max |action diff| over the first 20 steps of 10
rollouts between parity mode and fast mode, as a sanity number.

Labeler parity. On 1,000 saved states: C++ distances within 1e-6 m of the Python
implementation; identical nearest-pair identities; the analytic test cases from the original
validation pass. For the CUDA variant: within 1e-5 m of the C++ version, and report any case
where the nearest pair differs because of ties.

Training parity (only if the optional training milestone runs): loss curve at the same global
batch and seed within noise of a single-GPU run over 500 steps.

Every milestone's PR includes its parity results in the description.

---

## 7. Milestones

Each milestone: problem, baseline to measure, what to build, correctness check, expected gain
(GUESS until measured), time, skill proved, the one number for the resume.

### M1. Profile and baseline (2 days)
Problem: nobody knows where eval time goes. Build: section 4. Check: numbers reproduce on a
second run within 10%. Gain: none; this is the ruler. Skill: profiling (Nsight Systems,
torch.profiler). Number: "X% of step time in simulator/labeler, Y% in policy, GPU util Z%".

### M2. C++ clearance labeler (3-4 days)
Problem: the Python GJK labeler is reported at ~0.5 s per step, likely the single largest
CPU cost. Baseline: M1's labeler microbenchmark. Build: C++17 GJK (port the existing algorithm
one to one first; do not change the maths), hull cache, bounding-sphere pruning, thread pool
over hull pairs, pybind11 bindings, same return values as the Python monitor; a Python fallback
flag. Check: labeler parity in section 6, plus the analytic cases. Gain: GUESS 20-100x per
call. Time: 3-4 days. Skill: C++ performance, pybind11, parity testing. Number: "labeler from
A ms to B ms per step; eval wall-clock from C to D with no other change".

### M3. Parallel simulators + batched policy server (1 week)
Problem: one environment at a time; GPU waits on MuJoCo and MuJoCo waits on the GPU.
Baseline: M1 end-to-end rollouts per hour. Build: worker pool (start with multiprocessing,
N = number of physical cores minus 2), Triton server with the OFT Python backend and dynamic
batching, client stub, manifest runner, result aggregation, parity mode. Measure throughput
vs N and vs max batch delay (sweep 1, 5, 10, 20 ms). Check: Tier A on 10 starts, Tier B on
the 60 and intrusion-5 sets. Gain: GUESS 4-8x rollouts per hour on one GPU. Skill: inference
serving (Triton), multiprocess infra, dynamic batching, API design. Number: "rollouts/hour
from X to Y at batch Z; GPU util from A% to B%".

### M4. Policy forward optimisation (2-3 days)
Problem: the 7B forward is called once per 8 steps; latency bounds the batch wait and the
per-worker step rate. Baseline: forward latency and GPU util from M1 at batch 1, 4, 8, 16.
Build, measured one at a time and kept only if they win: bf16 end to end (check what is
already bf16), `torch.compile` on the LLM trunk, CUDA graphs for the fixed-shape forward,
pre-tokenised instructions cached per task, image preprocessing moved to the server GPU
(resize/normalise as a batched op). Check: Tier B after each change; max |action diff| vs
parity mode reported. Gain: GUESS 1.5-3x forward latency. Skill: GPU inference optimisation.
Number: "forward latency from X ms to Y ms at batch 8; throughput +Z%".

### M5. Deployment packaging (3 days)
Problem: the harness runs only from SSH + tmux on one box. Build: Docker image for the Triton
server (model repository, config.pbtxt, health check), Docker image for workers, compose file
for the box, Kubernetes manifests (server Deployment with GPU resource request, worker Job
with parallelism N, a ConfigMap for the manifest), Prometheus scrape of Triton metrics, a
`make bench` target. Test the k8s manifests on kind/minikube with a CPU-only stub model if no
GPU cluster is available, and say exactly that in the README. Check: a fresh clone + `make
bench` reproduces the M3 numbers. Skill: containers, Kubernetes, microservices, metrics.
Number: "one-command reproducible benchmark; cold start X s".

### M6. One justified CUDA kernel (3-4 days, after M2 numbers exist)
Problem: with N workers the labeler runs N times in parallel on the CPU; is a batched GPU
GJK worth it? Build: a CUDA kernel that takes a batch of (robot hull, obstacle hull) pairs
across all workers' current states and returns per-pair distances (GJK is branchy; a
simpler first version computes support-function-based distance for convex hulls with a fixed
iteration cap). Profile with Nsight Compute (occupancy, memory throughput). Compare against
the C++ threaded version at batch sizes 1, 8, 32, 128 and report the crossover. Keep whichever
wins in production; keep both in the repo. Check: labeler parity (CUDA vs C++). Gain: unknown;
the honest outcome may be "C++ wins below 64 pairs". Skill: CUDA, Nsight Compute, knowing when
not to use the GPU. Number: "batched GJK at X us/pair on GPU vs Y on 16 threads; crossover at
Z pairs".

### M7. Write-up and release (2 days)
README as a developer tutorial: what it is, how to run, the benchmark table (every number from
`bench/`), the correctness section, the two scoring-bug fixes with a before/after on the
published numbers, limitations. A short blog-style post for Varun's site or Medium. License
and attribution for LIBERO-Safety, OpenVLA-OFT, openpi. Open-source only after Heng's OK on
which parts (the harness and labeler are ours; model weights and the dataset are not
redistributed).

Optional M8, only with a second GPU for a weekend: DDP fine-tune of the continued-LoRA
recipe on 2 GPUs with the TFDS data path kept busy (prefetch, parallel decode); loss parity
vs single GPU at the same global batch; samples/s and MFU. Skill: multi-GPU training basics.
Do not start without the hardware.

### 7a. Optional capstone: a real motion planner for the demo generator (1-2 weeks)

Why it is here. The research closed because the scripted pick-lift-place planner could not
produce demonstrations at the obstacle placements where the policy actually collides (1 scene
in 197 on oa1; the obstacle sits in the carried bowl's descent to the plate and the way through
is to approach the plate from another side). A sampling-based or GPU planner with whole-arm
collision checking is the untested route, and it is also the most NVIDIA-relevant single
milestone in this plan.

Build. Replace the waypoint planner inside `datagen/dgcore.py` with either RRT-Connect from
OMPL (CPU, simplest to get right) or cuRobo (NVIDIA, GPU, batched; the stronger resume line),
planning in joint space with the carried object attached to the gripper and every robot link
checked against every obstacle hull (the elbow link `robot0_link5` is the one the scripted
planner could not see). Keep everything downstream unchanged: the OSC executor, the 2 cm GJK
certification, the rendered-replay gate, the LeRobot v2.1 writer, the TFDS converter.

Measure. Acceptance on the same 197 eval-rule oa1 scenes (baseline 1/197 MEASURED); time per
plan; clearance distribution of accepted demos. Then, if acceptance is usable, the existing
fine-tune and re-test pipeline runs unchanged (one day) and answers the research question the
project could not: does data placed on the policy's trajectory teach avoidance?

Number: "planner acceptance on collision scenes from 0.5% to X% at Y s per plan; Z certified
demos; fine-tune re-test collisions A to B". Skill: motion planning, collision checking, cuRobo
if chosen. This can replace M6 if the audit shows a CUDA GJK kernel is not justified; one of
the two is enough.

### 7b. What makes it resume-worthy beyond the code

- A public repo under Varun's name that looks maintained: README with the benchmark table at
  the top, install in three commands, `make bench` reproduces every number, tests in CI (the
  parity suite at minimum), tagged releases, clean per-milestone commit history.
- Every number with provenance: the script, the card, the driver, the commit. "3.4x on one
  A6000, run `bench/e2e.py`" beats "10x".
- One write-up (blog post or long README section): the problem, the profile that showed where
  time went, what was built, what was measured, what did not work. The CUDA kernel losing to
  C++ is a good section if that is what happens.
- A 30-second GIF or clip of N simulators against one batched server with the throughput
  counter.
- Proof of use: the lab running its next evaluation on the harness; Heng in the
  acknowledgements. Ask for both when asking about open-sourcing.
- A file of interview stories kept as they happen (`docs/stories.md`): the replay bug, the
  gripper mapping bug, the labeler that checked the gripper and not the elbow, the number
  reported as measured that was assumed. Each is a two-minute answer about measuring before
  trusting.
- A clear stopping point. Five finished milestones beat eight half-done ones; anything not
  done by the deadline is cut and listed as future work.

---

## 8. Benchmark table (README template; fill from `bench/`)

| metric | before (current driver) | after | how measured |
|---|---|---|---|
| rollouts per hour, 60-start eval, 1 GPU | | | `bench/e2e.py` |
| per-step wall-clock breakdown (sim / render / forward / labeler / glue) | | | `bench/breakdown.py` |
| labeler ms per step (Python / C++ / CUDA) | | | `bench/labeler.py` |
| policy forward latency at batch 1 / 8 / 16 (ms) | | | `bench/forward.py` |
| GPU utilisation during eval (%) | | | `nvidia-smi dmon` |
| parity: max action diff, parity mode vs current driver | | | `bench/parity.py` |
| statistical: collisions / completions on 59 starts, golden vs fast | | | `bench/equiv.py` |
| cold start of the server (s) | | | `bench/coldstart.py` |

---

## 9. Constraints and rules for the agent

- Shared box. GPU 0 belongs to someone else; never touch it, never kill processes you did not
  start. Check `nvidia-smi` and `who` before heavy runs. Keep disk under control; delete only
  your own intermediates.
- Never edit the existing drivers in place; the current `rollout_openvla_disp.py` is the
  regression reference and must keep running unchanged. New code lives under `fastvla/`.
  Reuse by import, not by copy, where the import is clean; otherwise copy with a comment
  naming the source and the reason.
- Every GPU-pass array that a later step depends on is saved before use.
- Measure before and after every change, on the same card, same seeds. No number goes into a
  README from memory.
- Determinism: same-card comparisons only; record the card, driver version, torch version and
  commit hash in every results file.
- Do not claim bitwise reproduction for batched or compiled modes; claim it only for parity
  mode and prove it.
- Author is Varun. No AI attribution in code, commits, docs or the README.
- Git: `fastvla/` is its own repository (not a subdirectory of the research repo), one commit
  per milestone step with a message that states the measured before/after. The remote is a
  private GitHub repo under Varun's account; push at the end of each milestone, never
  mid-change. It goes public only when Varun says so, after Heng's OK on the LIBERO-Safety
  parts. `.gitignore` excludes checkpoints, datasets, videos, traces, logs and conda envs; run
  a secret scan before every push (no W&B keys, no tokens). If `git push` needs credentials
  that are not configured on the box, stop and tell Varun; never paste a token into a prompt.
- Do not take a research session's context into this project. Start from
  `PROJECT_HANDOFF.md`, this plan, and the code; the research findings are background, not
  instructions.

---

## 10. Risks

- The audit may show the forward dominates (7B at ~50-100 ms per call is plausible). Then
  parallel workers help less than hoped and M4 moves before M3. The plan allows that.
- Triton's Python backend adds overhead per request; if it eats the batching gain, fall back to
  a plain PyTorch gRPC server and keep Triton as the packaged option. Report both.
- MuJoCo rendering with EGL in many processes can fight over the GPU; measure worker scaling
  early (N = 1, 2, 4, 8, 16) and find the knee.
- bf16 and compile may shift the one marginal completion (oa6_L1 init 6) and others; Tier B
  tolerances exist for this, but if the transition matrix is not near-diagonal, stop and find
  out why before claiming speed.
- GJK in CUDA is branchy; a naive port may be slower than C++. That is an acceptable finding if
  measured and explained.
- Heng's approval is needed before open-sourcing anything that includes LIBERO-Safety code or
  the fixed scoring; ask early.

---

## 11. Resume bullets (placeholders until measured)

- Built FastVLA-Eval, a batched inference and parallel-simulation harness for evaluating 7B
  vision-language-action policies on LIBERO-Safety: X workers + NVIDIA Triton dynamic batching
  raised throughput from A to B rollouts/hour on one GPU (Cx) with statistically identical
  success and collision rates (binomial check on 60 seeded starts).
- Ported the GJK convex-clearance labeler to multithreaded C++ (pybind11), cutting per-step
  cost from A ms to B ms with 1e-6 m parity against the Python reference on 1,000 states.
- Optimised the policy forward with bf16, torch.compile and CUDA graphs: latency A to B ms at
  batch 8, GPU utilisation A% to B%, action parity reported.
- Wrote a batched CUDA GJK kernel and profiled it with Nsight Compute; documented the crossover
  (GPU wins above N pairs) and kept the faster path in production.
- Containerised the harness (Docker, Kubernetes manifests, Triton metrics to Prometheus) with a
  one-command reproducible benchmark; open-sourced with the two LIBERO-Safety scoring fixes.

---

## 12. First prompt to give the building agent

"This is a new project, separate from the research. Read PROJECT_HANDOFF.md, then
vla_eval_harness_plan.md in full, before doing anything. Do milestone 0 (section 4, the audit)
and nothing else. Create fastvla/ with its own git repo (check git user.name and user.email
are Varun's; no AI attribution). Everything you write goes under fastvla/audit/. Do not modify
any existing file. GPU 1 only; check nvidia-smi and who first. Run every GPU job under
run_with_alert.py with RUN_LOG_NAME=M0_AUDIT_<table>. Save every profile and timing array to
disk before summarising. Check which nsys, ncu and torch.profiler exist; if nsys is missing say
so in AUDIT.md and use torch.profiler plus manual timers. Table 1 must reproduce the GPU 1
reference (29/60 collisions, 26/60 completions on 17500) or the audit is invalid: stop and
report. Then list every single-environment assumption in the current driver and write the
M3-vs-M4 ordering decision with evidence. Commit AUDIT.md and the audit scripts as the first
commit. Delete nothing. Stop and report with the five tables pasted in full."

---

## 13. Still to specify (Varun, before the first prompt)

Varun said on 13 Oct that he wants to add context about how the project should be built before
the first prompt goes out. Slots this plan leaves open for that:

- Repo name and whether `fastvla/` lives inside `behavior/displacement/` or beside it.
- Triton in v1, or a plain PyTorch gRPC server first with Triton as the packaged option.
- Which of M6 (CUDA GJK) and 7a (motion planner) to keep if time allows only one.
- Hard deadline for the whole project (drives the stopping point in 7b).
- Whether the harness must also serve Pi0.5 in v1 or OpenVLA-OFT only.
- Coding standards (Python version, formatter, test framework), if he has a preference.
- Anything from his own build experience (Adobe backend, the IC-memo RAG service) he wants
  reflected in the service design.
