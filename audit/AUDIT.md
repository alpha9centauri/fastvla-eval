# M0 audit — baseline and profile

FastVLA-Eval milestone 0 (plan section 4). Measures where closed-loop evaluation time
goes on the current driver, before any code is written. Nothing in this audit modifies
the research tree; the driver, the 17500 checkpoint and every reference output file are
untouched.

Run 2026-10-14 by Varun Chandra on gpu5, GPU 1 only.

Every number is tagged **MEASURED** (produced by a script in this directory during this
audit) or **ESTIMATE** (derived from an artifact that already existed). The script and the
saved artifact are named in each table. Raw timing arrays, profiles and traces are on disk
under `audit/arrays/` and `audit/traces/` and are deliberately not committed.

## Provenance

| field | value |
|---|---|
| card | NVIDIA RTX A5000 (GPU 1 of 2; GPU 0 belongs to another user and was not touched) |
| nvidia driver | 530.30.02 (2 x RTX A5000, 24564 MiB each) |
| torch | 2.2.0+cu121 (cuda 12.1, cudnn 8902) |
| numpy | 1.26.4 |
| conda env | `libero-vla` (python 3.10.20) — every rollout; `libero-vla-finetune` for the pandas-only trace analysis |
| driver | `rollout_openvla_disp.py` md5 `a3bd13daa60d30474eec023827560e03`, unmodified |
| checkpoint | `…--headinit--17500_chkpt` (15 G), loaded merged-direct |
| profilers | nsys 2023.1.2.43, ncu 2023.1.0.0, torch.profiler (torch 2.2.0) — all present |
| `kernel.perf_event_paranoid` | 3 — so `ncu` CPU sampling and some metric sets need elevation that is unavailable. GPU kernel metrics and `nsys` CUDA/OSRT tracing work; nsys was run with `--sample=none --backtrace=none`. |

Two documented deviations from the handoff's pasted reference command, both in
`t1_e2e.sh`: `OUTDIR` points at `audit/t1_out` so no reference file is touched, and
`MUJOCO_EGL_DEVICE_ID=1` is added. The second matters — EGL device enumeration ignores
`CUDA_VISIBLE_DEVICES`, so without it MuJoCo renders on EGL device 0, which is the other
user's GPU. The handoff's own hard rule requires the variable; the pasted command omits it.

## Table 1 — end to end (MEASURED)

`t1_e2e.sh` + `t1_summarize.py` → `arrays/t1_e2e.json`, run log `audit/M0_AUDIT_T1`.
One full 60-start evaluation of 17500, no displacement, current driver, under
`run_with_alert.py` with `RUN_LOG_NAME=M0_AUDIT_T1`.

| metric | value |
|---|---|
| rollouts | 60 (6 tasks x 10 inits) |
| **collisions** | **29/60** (reference 29/60) |
| **completions** | **26/60** (reference 25 or 26) |
| validity gate | **PASS** |
| wall-clock, rollout loop | 3032 s = 0.84 h |
| wall-clock, incl. model load | 3066 s (startup 34 s) |
| **rollouts per hour** | **71.24** (loop only); 70.45 incl. startup |
| seconds per rollout | 50.5 |
| total control steps | 15935 (mean 265.6/rollout, 34 hit the 300 cap) |
| **control steps per second** | **5.26** |
| ms per control step | 190.3 |
| **GPU 1 SM utilisation** | **mean 9.9%**, p50 0.0%, p95 69.0% over 3044 1 Hz samples |
| GPU 0 SM utilisation during the run | mean 0.0% — not ours, sampled only to show it was left alone |

Per task:

| task | rollouts | collisions | completions | control steps |
|---|---|---|---|---|
| oa1 | 10 | 0 | 9 | 2186 |
| oa2 | 10 | 9 | 4 | 2779 |
| oa3 | 10 | 6 | 0 | 3000 |
| oa4 | 10 | 5 | 3 | 2920 |
| oa5 | 10 | 9 | 0 | 3000 |
| oa6_L1 | 10 | 0 | 10 | 2050 |

## Table 2 — per-step breakdown (MEASURED)

`N_INIT=1 t2_run.sh timers 6` → `t2_breakdown.py` → `arrays/t2_alltasks.{json,npz}`.
One rollout on each of the six tasks, so the mix matches Table 1 rather than oa1 alone.
The driver runs unmodified; `profile_hooks.py` wraps its components by monkeypatch
before runpy executes it. The monitor hooks `sim.step`, so monitor and render time are
nested inside `env.step`; the controller/obs-assembly row is `env.step` minus the three
nested rows and is therefore derived, not timed directly.

1575 control steps over 6 rollouts. Accounted wall-clock 314.5 s = 199.7 ms per control step.

| component | ms/step mean | ms/step p95 | share of step | calls |
|---|---|---|---|---|
| MuJoCo physics (`sim.step` x25 substeps) | 4.50 | 8.54 | 2.3% | 40875 |
| camera rendering (2 x 256x256) | 1.21 | 1.51 | 0.6% | 3330 |
| SafetyMonitor total (contacts + clearance) | 142.11 | 348.25 | 71.2% | 40875 |
| &nbsp;&nbsp;of which GJK clearance | 134.10 | 342.64 | 67.2% | 3150 |
| OSC controller + obs assembly (derived) | 17.86 | 28.47 | 8.9% | derived |
| obs preprocessing (`prepare_observation`) | 8.60 | 10.97 | 4.3% | 1575 |
| **policy forward (`get_action`)** | 25.37 | 186.04 | 12.7% | 200 |
| `process_action` | 0.02 | 0.03 | 0.0% | 1575 |
| **total (`env.step` + prep + forward + process)** | **199.68** | 428.59 | 100% | |

The forward runs once per 8-step action chunk: 200 calls over 1575 control steps.
Per **call** it is 199.82 ms; amortised per **control step** it is
25.37 ms, which is the 12.7% share above.

Hull pairs evaluated exactly per clearance call: mean 122.9, p95 591, max 1334, over 3150 calls.

GPU side, `torch.profiler` over a 12-forward window (`t2_run.sh torch 1`, `arrays/t2_torch_torchprof_cuda.txt`, chrome trace saved): self CUDA 3673.7 ms vs self CPU 17472.2 ms across 175 op types. CUDA activity tracing inflates CPU time, so the table above is from the timers-only run.

`cProfile` over one rollout is saved as `arrays/t2_cprofile.prof` (+ `_cprofile.txt`, cumulative and tottime views) for the Python-level call counts.

## Table 3 — Nsight Systems trace, GPU idle gaps (MEASURED)

`t3_nsys.sh` → `traces/t3_rollout.nsys-rep` + `nsys stats` CSVs → `t3_gaps.py` →
`arrays/t3_gaps.json`. One rollout. Kernel intervals are merged, then the holes between
them are the device-idle gaps.

| metric | value |
|---|---|
| traced window | 66.41 s |
| kernel launches traced | 86693 |
| **GPU busy** | **9.8%** (6.49 s) |
| **GPU idle** | **90.2%** (59.92 s) |
| idle gaps | 86692 |
| gap length mean / p50 / p95 / max | 0.69 / 0.00 / 0.05 / 6634.96 ms |

Most of those 86k gaps are micro-holes between consecutive kernels inside one forward
(p50 1.2 us), which is not the question. The macro gaps — windows where the device has
nothing at all to do — are these:

| macro gap threshold | count | total | mean | p50 | max | share of window |
|---|---|---|---|---|---|---|
| > 10 ms | 41 | 58.2 s | 1420 ms | 1497 ms | 6635 ms | 87.6% |
| > 50 ms | 37 | 58.1 s | 1571 ms | 1690 ms | 6635 ms | 87.5% |
| > 50 ms, excluding the one startup gap | 36 | | 1431 ms | 1652 ms | 2394 ms | |

**Answer to the plan's question.** Yes, and by a lot. The GPU is idle
90.2% of the traced window, and 87.5% of it sits in just
37 gaps averaging 1571 ms each. That count is not a coincidence:
a 300-step rollout re-plans every 8 steps, so there are 300/8 = 37 forward passes and
therefore 37 windows in which the process is stepping MuJoCo and running the GJK labeler
with nothing resident on the device. The gap length is the independent check — 8 control
steps at Table 2's 199.7 ms/step is 1598 ms, against a measured mean of
1571 ms.

## Table 4 — clearance labeler microbenchmark (MEASURED)

`t4_run.sh` → `t4_labeler.py` → `arrays/t4_labeler.json` + per-call arrays.
Rollouts replayed from the actions Table 1 saved, so no policy and no GPU compute is
involved. `distmax` 0.5 is the production cutoff (`physics_safety.DEFAULT_DISTMAX`);
the tighter values test the plan's claim that a wide cutoff defeats the pruning.

| distmax (m) | calls | ms/call mean | ms/call p95 | exact hull pairs/call mean | us per exact pair | ms per control step |
|---|---|---|---|---|---|---|
| 0.5 **(production)** | 1000 | 64.12 | 161.42 | 101.1 | 634 | 128.25 |
| 0.1 | 1000 | 45.40 | 103.21 | 59.7 | 760 | 90.80 |
| 0.05 | 1000 | 24.19 | 41.69 | 0.0 | n/a | 48.38 |

Obstacle hull pieces per task: {"oa1": 248}.

The last column is `ms/call x 2`, because the monitor hooks `sim.step` and so runs per
MuJoCo substep: with `clearance_every=25` and 25 substeps per control step, clearance is
computed on substep 1 and substep 25 — twice per control step, not once.

## Table 5 — fine-tune data path (ESTIMATE, deliberately not re-measured)

`t5_finetune.py` → `arrays/t5_finetune.json`, parsed from
`src/displacement/ft_liftonly.log` (a complete 3000-iteration run, 8-9 Oct 2026).

| metric | value |
|---|---|
| recipe | batch 1 x grad-accum 8 (effective 8), lr 5e-5, LoRA r32/a16 |
| s/iteration, median of 3015 tqdm samples | 4.84 |
| s/iteration from total wall | 5.31 (3000 steps in 15930 s) |
| **samples/s (median)** | **1.65** |
| samples/s (from total wall, includes checkpoint saves) | 1.51 |
| 200 steps would take | 17.7 min |
| GPU utilisation during training | **NOT MEASURED** |

Why not re-measured on GPU 1. `datagen/run_ft_liftonly.sh` passes the reference 17500
checkpoint as `--vla_path`, and `finetune_headinit.py` rewrites `config.json` **inside**
that directory — `ft_liftonly.log` lines 18-20 record it creating
`config.json.back.20261008_221207` and then updating `config.json`. That directory is the
base policy for every parity comparison in the plan, and this project may not modify
anything outside `fastvla-eval`, so a fresh 200-step run was not started. The existing log
is also the better sample: 3000 iterations on the same card rather than 200. The cost is
that training-time GPU utilisation stays unknown; it is only needed for the optional M8
(two-GPU DDP), which the plan says not to begin without a second card.

## Cross-validation

Four numbers were measured twice by unrelated means. They agree, which is the reason to
trust the rest of the audit.

| quantity | method A | method B |
|---|---|---|
| GPU utilisation during eval | `nvidia-smi dmon`, 1 Hz over the full 60-start run: **9.9%** | nsys kernel intervals over one rollout: **9.8%** |
| time between forward passes | nsys macro gap mean: **1571 ms** | 8 control steps x Table 2's per-step total: **1597 ms** |
| GJK clearance cost per control step | Table 4 replay, 2 calls x 64.12 ms: **128.2 ms** | Table 2 in-loop timer: **134.10 ms** |
| per-control-step wall-clock | Table 1, uninstrumented, 15935 steps: **190.3 ms** | Table 2, instrumented, 6 tasks: **199.7 ms** (+4.9% instrumentation) |

## The M3-vs-M4 ordering decision

The plan's rule (section 4): *if the policy forward is under 30% of step time and the
simulator plus labeler dominate, M3 is the big win and M4 is small; if the forward
dominates, reverse the order of 3 and 4.*

**Evidence.** The policy forward is **12.7%** of step time amortised (200 ms per call, once per
8-step chunk). The simulator plus labeler are **74.0%**, of which the
SafetyMonitor alone is **71.2%**. The GPU is idle 90.2% of the time.

**Decision: keep the plan's order — M2, then M3, then M4.** The forward is far below the
30% threshold, so M4 cannot be the big win: even making the forward free would remove
only 12.7% of step time. M3 keeps its place ahead of M4.

**One correction to the plan's framing.** The plan calls M2 a 3-4 day warm-up and M3 the
big win. The measurement says M2 is the biggest single win available: the labeler is
71.2% of step time on its own, and it is pure CPU Python. cProfile over one
rollout (`arrays/t2_cprofile_cprofile.txt`) puts the whole cost in two functions —
`convex_distance._closest_on_simplex` at 269,650 calls and 15.1 s of self time, and
`gjk_distance` at 81,068 calls and 6.6 s, with 590,806 `numpy.linalg.solve` calls
underneath them. That is the GJK simplex solver running in Python with a 3x3 LAPACK
solve per iteration, which is exactly what a C++ port removes.

Two further findings that change how M2 and M3 should be built:

1. **The labeler runs twice per control step, not once.** `SafetyMonitor.attach()` hooks
   `sim.step`, so it fires per MuJoCo substep; with `clearance_every=25` and 25 substeps
   per control step, clearance is computed on substep 1 and substep 25. Any speedup claim
   must use that factor of 2.
2. **Pruning already works; the cutoff is what costs.** Table 4's sweep shows the exact-pair
   count falling from 101.1 per call at the production `distmax=0.5` to **zero** at 0.05,
   where every candidate is pruned before any exact GJK call — and yet the call still costs
   24.19 ms. So roughly 24 ms of the 64 ms is the per-actor union-hull GJK and the
   bounding-volume work, which pruning cannot remove and which a C++ port must also cover.
   M2 should not expect the plan's 20-100x from threading the pair loop alone.

## Estimates in the plan that this audit replaces

The plan's instruction is that every number in it not marked MEASURED is an estimate and
must be replaced by a measurement. These are the ones milestone 0 can settle.

| plan's figure (section) | plan said | measured | verdict |
|---|---|---|---|
| 60 starts for one arm, wall-clock (3.4) | "on the order of 1-2 hours on GPU 1" | **51 min** (71.2 rollouts/h) | faster than estimated; the low end was right |
| labeler cost per step (3.3) | "about 543 ms per step with a wide cutoff", explicitly NOT independently measured | **134.10 ms/step** in-loop, **128.2 ms/step** on replay | the real figure is ~4.0x smaller than the plan's number |
| per-step split (3.4) | "the split is NOT measured" | labeler 71.2%, forward 12.7%, controller+obs 8.9%, prep 4.3%, physics 2.3%, render 0.6% | measured, Table 2 |
| forward latency (10) | "7B at ~50-100 ms per call is plausible" | **200 ms per call** at batch 1 | 2-4x slower than the guess; two 256x256 images plus a ~57-token action span |
| oa1 obstacle hull pieces (3.3) | "the oa1 bottle is 244 hull pieces" | **248** collision geoms across the declared obstacle | close; 244 was about right |
| fine-tune iteration time (4.5) | "~4.8 s/it at effective batch 8" | **4.84 s/it** median, 5.31 s/it from total wall | the estimate was accurate |

Two the audit does **not** settle, and why:

- **Rendering looks too cheap.** 1.21 ms/step for two 256x256 camera renders is suspiciously
  low. `MjSim.render` is timed, but robosuite may be doing part of the work elsewhere in
  `_get_observations`, which would land in the derived controller/obs row (17.86 ms/step)
  instead. Before M3 leans on "rendering is free", that row should be split properly.
  Flagged rather than claimed.
- **Training GPU utilisation**, for the reason in Table 5.

---

## Addendum, 2026-10-14 — the rendering figure, split (MEASURED)

Table 2 reported `MjSim.render` at 1.21 ms per control step for two 256x256 cameras and
flagged it as suspiciously cheap, on the suspicion that robosuite did part of the render
work elsewhere and it was hiding in the derived controller/obs row. M3 depends on knowing
which, so it was measured: `t2b_render_split.py` / `t2b_run.sh`, 100 control steps on
oa1 init 0, artefacts `arrays/t2b_render_split{,_clean}.json`.

**The flag is resolved: rendering really is that cheap, and nothing was hiding.** What was
actually in the derived row is forward kinematics and the OSC controller.

robosuite's `MujocoEnv.step` runs, per control step:

```
for i in range(control_timestep / model_timestep):   # 25 substeps
    sim.forward()            # extra forward kinematics
    _pre_action(action)      # OSC controller (calls sim.forward() again itself)
    sim.step()               # physics; SafetyMonitor is hooked onto this
    _update_observables()    # sensors, including both cameras
_post_action(action); _get_observations()
```

| component of `env.step` | ms/step | p95 | share |
|---|---|---|---|
| SafetyMonitor (contacts + GJK) | 127.990 | 206.082 | 77.0% |
| OSC controller `_pre_action` x25 | 13.923 | 14.560 | 8.4% |
| &nbsp;&nbsp;its nested `sim.forward()` | 7.810 | 8.343 |  |
| `sim.forward()` x25, robosuite's own loop | 8.517 | 9.533 | 5.1% |
| MuJoCo physics `sim.step()` x25 | 8.240 | 8.976 | 5.0% |
| `_update_observables` x25 (sensor update) | 2.157 | 2.321 | 1.3% |
| &nbsp;&nbsp;of which `MjSim.render` x2 (the cameras) | 1.393 | 1.514 |  |
| `_get_observations` x1 (assemble the dict) | 0.147 | 0.181 | 0.1% |
| `_post_action` x1 | 0.652 | 0.708 | 0.4% |
| unaccounted residual (timer overhead, loop glue) | 4.534 | | 2.7% |
| **`env.step` total** | **166.161** | | 100% |

**Real render cost: 0.696 ms per 256x256 EGL camera render**, 1.393 ms/step for both.
The whole observation path — sensor update plus dict assembly — is 2.304 ms/step, 1.4% of `env.step`.

Two corrections this forces, both relevant to M3:

1. **Rendering is not a scaling risk on this box.** The plan's risk list expects EGL
   rendering in many processes to fight over the GPU. At 0.7 ms per camera it is 1.4% of
   a control step, so the worker-scaling knee will be set by the labeler and the
   controller, not by rendering. Worth re-checking at N=8 and N=16, but it is not the
   thing to design around.
2. **`sim.forward()` costs more than `sim.step()`.** It is called twice per substep —
   once by robosuite's loop (8.52 ms/step) and once inside the OSC
   controller (7.81 ms/step) — for 16.33 ms/step against
   8.24 ms/step for the physics itself. Simulator cost is forward
   kinematics, not integration. Nothing in M2 touches it; M3 should know it is there.

The measurement needed two corrections of its own, both recorded in the artefact's
`note`: the first timing bucket was dropped because it holds the driver's 10 `WAIT` dummy
steps, and `_pre_action` had to be patched on `RobotEnv` rather than `MujocoEnv`, whose
definition is shadowed. `sim.forward` is attributed by caller with a re-entrancy flag;
without that it is charged twice and the residual goes negative.

---

## Single-environment assumptions in `rollout_openvla_disp.py`

Line numbers are against the driver at md5 `a3bd13daa60d30474eec023827560e03`
(`/data_new/vchandr2/vla-safety-interp/src/displacement/rollout_openvla_disp.py`, 671 lines).
M3 replaces this loop with N worker processes against one batched server, so each of these has
to be broken deliberately. Grouped by what breaks if you simply run the file N times at once.

## A. Shared output files — concurrent runs corrupt each other

| # | assumption | lines | what breaks |
|---|---|---|---|
| A1 | One `OUTDIR` per process, and `_save()` rewrites every output file **in full** from in-process accumulators (`rows`, `trajs`, `acts`, `probes`, `clamps`) every 10 rollouts and once at the end | 54, 604-614, 664 | Two processes sharing an `OUTDIR` each write a *complete* `rows_*.json` containing only their own rollouts. Last writer wins; the other half is gone. Not a merge, a replacement. |
| A2 | Output names are fixed per stage, not per worker: `ROWS_JSON`, `CLAMPS_JSON`, `TRAJ_NPZ`, `ACT_NPZ`, `PROBE_NPZ` | 69-71, 323-325 | Nothing in a name distinguishes worker, shard or attempt. The in-file comment at 66-68 records that one-filename-per-stage collisions already destroyed completed stages three times. |
| A3 | `baseline_<STEP>.npz` is read once at import into a module-level dict and then **read-modify-written** inside the rollout loop | 326-327, 640-644 | Classic lost update: each process holds a stale copy of the whole dict and rewrites the file from it. The `WRITE_BASELINE` / `STEER_K` guard at 640-643 exists because this already clobbered a run's reference trajectories. |
| A4 | Videos go to `os.path.join(HERE, "videos")` — `HERE` is the **driver's own directory**, not `OUTDIR` — named `oft{STEP}__{tag}__i{init}__{arm}.mp4` | 659-660 | Two things at once. (i) Video output escapes `OUTDIR` entirely and lands in the research repo, so pointing `OUTDIR` elsewhere does not redirect it. (ii) The name has no run or worker id, so re-running the same (step, task, init, arm) silently overwrites. Already live: `videos/oft17500__oa1__i00__base.mp4` exists among 27 files, and a `videos_backup_pre_ftretest/` directory exists because someone had to rescue them. |
| A5 | `[PROGRESS] done=N total=N` is a single process-global counter, and `run_with_alert.py` parses it from one child's stdout | 631, 662 | With N workers the wrapper sees N interleaved counters and reports whichever line arrived last. The denominator is also per-process. |

## B. Process-global state — unsafe to share a process between rollouts

| # | assumption | lines | what breaks |
|---|---|---|---|
| B1 | The 7B policy, proprio projector, action head and processor are built **at module import**, one set per process | 159-167 | ~16 GB of GPU memory per process (measured in Table 1). N workers in this shape is N x 16 GB and will not fit on a 24 GB card past N=1. This is the whole reason M3 needs a server rather than N replicas. |
| B2 | `cap = {}` is a module-level dict written by a forward hook that indexes **batch element 0 only** (`h[0, ...]`) | 180, 189, 206-209 | Hard batch-size-1 assumption inside the probe capture path. Any server-side batching makes `h[0]` the wrong request's hidden state. `PROBE=0` sidesteps it for eval, but the plan keeps the probe as a client of the harness. |
| B3 | The forward hook is registered on the **global** model object at import | 191, 211 | One hook per process, so per-rollout or per-request hook configuration is not expressible; and hooks accumulate if a module is re-imported. |
| B4 | Global RNG seeding: `random.seed(SEED); np.random.seed(SEED)` at import, re-seeded per rollout inside `run_one` | 40, 341 | Determinism depends on the process-global RNG being at a known point when a rollout starts. Interleaving work between rollouts in one process (as an async worker would) breaks the contract. The `noise` arm is fine — it uses a local `RandomState(1000+ii)` (416-417). |
| B5 | `FS_GLOBAL` and `PSTD` are module globals set in `main()` / memoised on first use, read inside `run_one` | 143-144, 148-153, 566 | Rollout work is not a pure function of its spec; it reads process state established elsewhere. A worker pool has to establish both before any rollout. |
| B6 | `baseline`, `VIDEOS`, `PR`, `STEER` are all import-time module globals derived from env vars | 125-136, 137-142, 169-177, 327 | Every per-rollout decision (does this one record video? does it steer?) is baked in at import, so one process cannot serve a heterogeneous manifest. |
| B7 | `sys.path` is mutated at import, and `rollout_bootstrap` **must** be imported before `openvla_utils` | 33, 36-38, 151 | Import order is a global side effect (the stub shim keeps TensorFlow out of inference). Any worker entry point has to reproduce it exactly; it cannot be a plain `import fastvla.sim.worker`. |

## C. Environment and device assumptions

| # | assumption | lines / source | what breaks |
|---|---|---|---|
| C1 | Exactly one `OffScreenRenderEnv` is alive at a time, created per rollout and closed on every exit path | 342, 366/387/401/405/473 | Five separate `env.close()` call sites; one missed path leaks an EGL context. With N workers each holding a context, a leak is N times worse and shows up as GPU memory, not as an error. |
| C2 | `MUJOCO_GL`, `MUJOCO_EGL_DEVICE_ID` and `CUDA_VISIBLE_DEVICES` are **process-level** env vars | handoff hard rules; `t1_e2e.sh` | Per-worker device placement has to be set at spawn, not configured in-process. Also the one real hazard found in this audit: EGL device enumeration ignores `CUDA_VISIBLE_DEVICES`, so an unset `MUJOCO_EGL_DEVICE_ID` renders on physical GPU 0 regardless of the CUDA mask — on this box, another user's card. |
| C3 | The policy forward happens **in the same process** as the simulator, synchronously, inside the control loop | 437-441 | The GPU idles while MuJoCo steps and MuJoCo idles while the GPU runs (quantified in Tables 2 and 3). Breaking this is M3's entire point. |
| C4 | `SafetyMonitor.attach()` monkeypatches `self.sim.step` on the **instance** and must be paired with `detach()` | `physics_safety.py` 158-175; driver 473 | Instance-level, so it is actually multiprocess-safe — worth recording as one assumption M3 does *not* have to fix. But an un-detached monitor keeps the env alive, so the pairing matters more with pooled workers. |

## D. Not an assumption, but a scaling fact M3 must plan around

The monitor hooks `sim.step`, so it runs per MuJoCo **substep**, not per control step: contacts every
substep, GJK clearance every `clearance_every` substeps (25 in production) plus substep 1
(`physics_safety.py` 333-335). At 25 substeps per control step that is 2 clearance calls and 25
contact scans per control step. Any per-step cost model for the labeler must use that factor, and
the C++ port in M2 inherits it.

