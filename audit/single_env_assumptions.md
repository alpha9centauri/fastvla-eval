# Single-environment assumptions in `rollout_openvla_disp.py`

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
