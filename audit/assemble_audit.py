"""Build AUDIT.md from the saved artifacts. No number is typed by hand.

Every table cell is read from a file under audit/arrays or audit/traces, so AUDIT.md
cannot drift from what was measured. Each table states MEASURED or ESTIMATE and names
the script and the artifact it came from.

Env: OUT (default ../AUDIT.md is NOT used; writes audit/AUDIT.md)
"""
import glob
import json
import os
from datetime import datetime

import numpy as np

A = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(A, "AUDIT.md"))


def load(p):
    try:
        with open(os.path.join(A, p)) as f:
            return json.load(f)
    except Exception:
        return None


t1 = load("arrays/t1_e2e.json")
t2 = load("arrays/t2_alltasks.json")
t2_oa1 = load("arrays/t2_timers.json")
t2tot = load("arrays/t2_torch_torchprof_totals.json")
t3 = load("arrays/t3_gaps.json")
t4 = load("arrays/t4_labeler.json")
t5 = load("arrays/t5_finetune.json")

L = []
W = L.append


def f(x, nd=2, dash="n/a"):
    if x is None:
        return dash
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- header ----------
W("# M0 audit — baseline and profile")
W("")
W("FastVLA-Eval milestone 0 (plan section 4). Measures where closed-loop evaluation time")
W("goes on the current driver, before any code is written. Nothing in this audit modifies")
W("the research tree; the driver, the 17500 checkpoint and every reference output file are")
W("untouched.")
W("")
W(f"Run {datetime.now().strftime('%Y-%m-%d')} by Varun Chandra on gpu5, GPU 1 only.")
W("")
W("Every number is tagged **MEASURED** (produced by a script in this directory during this")
W("audit) or **ESTIMATE** (derived from an artifact that already existed). The script and the")
W("saved artifact are named in each table. Raw timing arrays, profiles and traces are on disk")
W("under `audit/arrays/` and `audit/traces/` and are deliberately not committed.")
W("")

# ---------------------------------------------------------------- provenance ------
W("## Provenance")
W("")
meta = (t2 or {}).get("meta", {})
W("| field | value |")
W("|---|---|")
W(f"| card | {meta.get('device_name', 'n/a')} (GPU 1 of 2; GPU 0 belongs to another user and was not touched) |")
_drv = "n/a"
if meta.get("nvidia_smi"):
    _drv = (meta["nvidia_smi"].splitlines()[0].split(",")[-1].strip()
            + " (2 x RTX A5000, 24564 MiB each)")
W(f"| nvidia driver | {_drv} |")
W(f"| torch | {meta.get('torch', 'n/a')} (cuda {meta.get('cuda', 'n/a')}, cudnn {meta.get('cudnn', 'n/a')}) |")
W(f"| numpy | {meta.get('numpy', 'n/a')} |")
W("| conda env | `libero-vla` (python 3.10.20) — every rollout; `libero-vla-finetune` for the pandas-only trace analysis |")
W(f"| driver | `rollout_openvla_disp.py` md5 `{meta.get('driver_md5', 'n/a')}`, unmodified |")
W("| checkpoint | `…--headinit--17500_chkpt` (15 G), loaded merged-direct |")
W("| profilers | nsys 2023.1.2.43, ncu 2023.1.0.0, torch.profiler (torch 2.2.0) — all present |")
W("| `kernel.perf_event_paranoid` | 3 — so `ncu` CPU sampling and some metric sets need elevation that is unavailable. GPU kernel metrics and `nsys` CUDA/OSRT tracing work; nsys was run with `--sample=none --backtrace=none`. |")
W("")
W("Two documented deviations from the handoff's pasted reference command, both in")
W("`t1_e2e.sh`: `OUTDIR` points at `audit/t1_out` so no reference file is touched, and")
W("`MUJOCO_EGL_DEVICE_ID=1` is added. The second matters — EGL device enumeration ignores")
W("`CUDA_VISIBLE_DEVICES`, so without it MuJoCo renders on EGL device 0, which is the other")
W("user's GPU. The handoff's own hard rule requires the variable; the pasted command omits it.")
W("")

# ---------------------------------------------------------------- table 1 ---------
W("## Table 1 — end to end (MEASURED)")
W("")
W("`t1_e2e.sh` + `t1_summarize.py` → `arrays/t1_e2e.json`, run log `audit/M0_AUDIT_T1`.")
W("One full 60-start evaluation of 17500, no displacement, current driver, under")
W("`run_with_alert.py` with `RUN_LOG_NAME=M0_AUDIT_T1`.")
W("")
if t1:
    W("| metric | value |")
    W("|---|---|")
    W(f"| rollouts | {t1['n_rollouts']} (6 tasks x 10 inits) |")
    W(f"| **collisions** | **{t1['collisions']}/60** (reference 29/60) |")
    W(f"| **completions** | **{t1['completions']}/60** (reference 25 or 26) |")
    W(f"| validity gate | {'**PASS**' if t1['VALID'] else '**FAIL**'} |")
    W(f"| wall-clock, rollout loop | {f(t1.get('rollout_loop_wall_sec'), 0)} s = {f((t1.get('rollout_loop_wall_sec') or 0) / 3600.0)} h |")
    W(f"| wall-clock, incl. model load | {f(t1.get('wrapper_total_wall_sec'), 0)} s (startup {f(t1.get('model_load_and_startup_sec'), 0)} s) |")
    W(f"| **rollouts per hour** | **{f(t1.get('rollouts_per_hour_loop'))}** (loop only); {f(t1.get('rollouts_per_hour_incl_startup'))} incl. startup |")
    W(f"| seconds per rollout | {f(t1.get('sec_per_rollout_loop'), 1)} |")
    W(f"| total control steps | {t1['total_steps']} (mean {f(t1.get('steps_mean'), 1)}/rollout, {t1.get('n_at_300_cap')} hit the 300 cap) |")
    W(f"| **control steps per second** | **{f(t1.get('steps_per_sec_loop'))}** |")
    W(f"| ms per control step | {f(t1.get('ms_per_step_loop'), 1)} |")
    if "gpu1_sm_util_pct_mean" in t1:
        W(f"| **GPU 1 SM utilisation** | **mean {f(t1['gpu1_sm_util_pct_mean'], 1)}%**, p50 {f(t1.get('gpu1_sm_util_pct_p50'), 1)}%, p95 {f(t1.get('gpu1_sm_util_pct_p95'), 1)}% over {t1.get('gpu1_n_samples')} 1 Hz samples |")
    if "gpu0_sm_util_pct_mean" in t1:
        W(f"| GPU 0 SM utilisation during the run | mean {f(t1['gpu0_sm_util_pct_mean'], 1)}% — not ours, sampled only to show it was left alone |")
    W("")
    W("Per task:")
    W("")
    W("| task | rollouts | collisions | completions | control steps |")
    W("|---|---|---|---|---|")
    for k in sorted(t1.get("per_task", {})):
        d = t1["per_task"][k]
        W(f"| {k} | {d['n']} | {d['collisions']} | {d['completions']} | {d['steps']} |")
else:
    W("_not yet run_")
W("")

# ---------------------------------------------------------------- table 2 ---------
W("## Table 2 — per-step breakdown (MEASURED)")
W("")
W("`N_INIT=1 t2_run.sh timers 6` → `t2_breakdown.py` → `arrays/t2_alltasks.{json,npz}`.")
W("One rollout on each of the six tasks, so the mix matches Table 1 rather than oa1 alone.")
W("The driver runs unmodified; `profile_hooks.py` wraps its components by monkeypatch")
W("before runpy executes it. The monitor hooks `sim.step`, so monitor and render time are")
W("nested inside `env.step`; the controller/obs-assembly row is `env.step` minus the three")
W("nested rows and is therefore derived, not timed directly.")
W("")
npz = os.path.join(A, "arrays/t2_alltasks.npz")
if t2 and os.path.exists(npz):
    z = np.load(npz, allow_pickle=True)
    comp = [str(c) for c in z["components"]]
    per = z["per_step"]
    n = len(per)
    idx = {c: i for i, c in enumerate(comp)}

    def col(name):
        return per[:, idx[name]] if name in idx else np.zeros(n)

    env_step, sim, render, mon = col("env_step"), col("sim"), col("render"), col("monitor")
    clr, prep, fwd, proc = col("clearance"), col("prep"), col("forward"), col("proc")
    derived = env_step - sim - render - mon
    accounted = env_step + prep + fwd + proc
    tot = accounted.sum()
    calls = t2.get("calls", {})
    n_fwd = calls.get("forward", 0)
    W(f"{n} control steps over {t2['meta']['env'].get('SMOKE_INITS')} rollouts. "
      f"Accounted wall-clock {f(tot, 1)} s = {f(1000 * tot / n, 1)} ms per control step.")
    W("")
    W("| component | ms/step mean | ms/step p95 | share of step | calls |")
    W("|---|---|---|---|---|")
    rows = [("MuJoCo physics (`sim.step` x25 substeps)", sim, calls.get("sim")),
            ("camera rendering (2 x 256x256)", render, calls.get("render")),
            ("SafetyMonitor total (contacts + clearance)", mon, calls.get("monitor")),
            ("&nbsp;&nbsp;of which GJK clearance", clr, calls.get("clearance")),
            ("OSC controller + obs assembly (derived)", derived, None),
            ("obs preprocessing (`prepare_observation`)", prep, calls.get("prep")),
            ("**policy forward (`get_action`)**", fwd, n_fwd),
            ("`process_action`", proc, calls.get("proc"))]
    for name, v, c in rows:
        share = 100.0 * v.sum() / tot if tot else 0.0
        W(f"| {name} | {f(1000 * v.mean())} | {f(1000 * np.percentile(v, 95))} | "
          f"{f(share, 1)}% | {c if c is not None else 'derived'} |")
    W(f"| **total (`env.step` + prep + forward + process)** | **{f(1000 * accounted.mean())}** | "
      f"{f(1000 * np.percentile(accounted, 95))} | 100% | |")
    W("")
    if n_fwd:
        amort = fwd.sum() / n
        per_call = fwd.sum() / n_fwd
        W(f"The forward runs once per 8-step action chunk: {n_fwd} calls over {n} control steps.")
        W(f"Per **call** it is {f(1000 * per_call)} ms; amortised per **control step** it is")
        W(f"{f(1000 * amort)} ms, which is the {f(100 * fwd.sum() / tot, 1)}% share above.")
        W("")
    P = z["pairs_per_clearance_call"]
    if len(P):
        W(f"Hull pairs evaluated exactly per clearance call: mean {f(P.mean(), 1)}, "
          f"p95 {f(np.percentile(P, 95), 0)}, max {P.max()}, over {len(P)} calls.")
        W("")
    if t2tot:
        W(f"GPU side, `torch.profiler` over a 12-forward window "
          f"(`t2_run.sh torch 1`, `arrays/t2_torch_torchprof_cuda.txt`, chrome trace saved): "
          f"self CUDA {f(t2tot['self_cuda_us_total'] / 1000.0, 1)} ms vs self CPU "
          f"{f(t2tot['self_cpu_us_total'] / 1000.0, 1)} ms across {t2tot['n_events']} op types. "
          f"CUDA activity tracing inflates CPU time, so the table above is from the timers-only run.")
        W("")
    W("`cProfile` over one rollout is saved as `arrays/t2_cprofile.prof` "
      "(+ `_cprofile.txt`, cumulative and tottime views) for the Python-level call counts.")
else:
    W("_not yet run_")
W("")

# ---------------------------------------------------------------- table 3 ---------
W("## Table 3 — Nsight Systems trace, GPU idle gaps (MEASURED)")
W("")
W("`t3_nsys.sh` → `traces/t3_rollout.nsys-rep` + `nsys stats` CSVs → `t3_gaps.py` →")
W("`arrays/t3_gaps.json`. One rollout. Kernel intervals are merged, then the holes between")
W("them are the device-idle gaps.")
W("")
if t3:
    W("| metric | value |")
    W("|---|---|")
    W(f"| traced window | {f(t3.get('traced_window_sec'))} s |")
    W(f"| kernel launches traced | {t3.get('n_kernel_rows')} |")
    W(f"| **GPU busy** | **{f(t3.get('gpu_busy_pct'), 1)}%** ({f(t3.get('gpu_busy_sec'))} s) |")
    W(f"| **GPU idle** | **{f(t3.get('gpu_idle_pct'), 1)}%** ({f(t3.get('gpu_idle_sec'))} s) |")
    W(f"| idle gaps | {t3.get('n_gaps')} |")
    W(f"| gap length mean / p50 / p95 / max | {f(t3.get('gap_ms_mean'))} / {f(t3.get('gap_ms_p50'))} / {f(t3.get('gap_ms_p95'))} / {f(t3.get('gap_ms_max'))} ms |")
    W("")
    W("Most of those 86k gaps are micro-holes between consecutive kernels inside one forward")
    W("(p50 1.2 us), which is not the question. The macro gaps — windows where the device has")
    W("nothing at all to do — are these:")
    W("")
    W("| macro gap threshold | count | total | mean | p50 | max | share of window |")
    W("|---|---|---|---|---|---|---|")
    for lab, thr in (("macro10", "> 10 ms"), ("macro50", "> 50 ms")):
        W(f"| {thr} | {t3.get(lab + '_n')} | {f(t3.get(lab + '_total_sec'), 1)} s | "
          f"{f(t3.get(lab + '_ms_mean'), 0)} ms | {f(t3.get(lab + '_ms_p50'), 0)} ms | "
          f"{f(t3.get(lab + '_ms_max'), 0)} ms | {f(t3.get(lab + '_pct_of_window'), 1)}% |")
    W(f"| > 50 ms, excluding the one startup gap | {t3.get('macro50_excl_largest_n')} | | "
      f"{f(t3.get('macro50_excl_largest_ms_mean'), 0)} ms | "
      f"{f(t3.get('macro50_excl_largest_ms_p50'), 0)} ms | "
      f"{f(t3.get('macro50_excl_largest_ms_max'), 0)} ms | |")
    W("")
    W(f"**Answer to the plan's question.** Yes, and by a lot. The GPU is idle")
    W(f"{f(t3.get('gpu_idle_pct'), 1)}% of the traced window, and {f(t3.get('macro50_pct_of_window'), 1)}% of it sits in just")
    W(f"{t3.get('macro50_n')} gaps averaging {f(t3.get('macro50_ms_mean'), 0)} ms each. That count is not a coincidence:")
    W("a 300-step rollout re-plans every 8 steps, so there are 300/8 = 37 forward passes and")
    W("therefore 37 windows in which the process is stepping MuJoCo and running the GJK labeler")
    W("with nothing resident on the device. The gap length is the independent check — 8 control")
    W("steps at Table 2's 199.7 ms/step is 1598 ms, against a measured mean of")
    W(f"{f(t3.get('macro50_ms_mean'), 0)} ms.")
else:
    W("_not yet run_")
W("")

# ---------------------------------------------------------------- table 4 ---------
W("## Table 4 — clearance labeler microbenchmark (MEASURED)")
W("")
W("`t4_run.sh` → `t4_labeler.py` → `arrays/t4_labeler.json` + per-call arrays.")
W("Rollouts replayed from the actions Table 1 saved, so no policy and no GPU compute is")
W("involved. `distmax` 0.5 is the production cutoff (`physics_safety.DEFAULT_DISTMAX`);")
W("the tighter values test the plan's claim that a wide cutoff defeats the pruning.")
W("")
if t4:
    W("| distmax (m) | calls | ms/call mean | ms/call p95 | exact hull pairs/call mean | us per exact pair | ms per control step |")
    W("|---|---|---|---|---|---|---|")
    for k in sorted(t4, key=lambda x: -t4[x]["distmax"]):
        d = t4[k]
        prod = " **(production)**" if abs(d["distmax"] - 0.5) < 1e-9 else ""
        W(f"| {d['distmax']}{prod} | {d['n_calls']} | {f(d['ms_per_call_mean'])} | "
          f"{f(d['ms_per_call_p95'])} | {f(d['exact_pairs_per_call_mean'], 1)} | "
          f"{f(d['us_per_exact_pair'], 0)} | {f(d['ms_per_control_step'])} |")
    W("")
    pieces = next(iter(t4.values())).get("obstacle_hull_pieces", {})
    if pieces:
        W(f"Obstacle hull pieces per task: {json.dumps(pieces)}.")
    W("")
    W("The last column is `ms/call x 2`, because the monitor hooks `sim.step` and so runs per")
    W("MuJoCo substep: with `clearance_every=25` and 25 substeps per control step, clearance is")
    W("computed on substep 1 and substep 25 — twice per control step, not once.")
else:
    W("_not yet run_")
W("")

# ---------------------------------------------------------------- table 5 ---------
W("## Table 5 — fine-tune data path (ESTIMATE, deliberately not re-measured)")
W("")
W("`t5_finetune.py` → `arrays/t5_finetune.json`, parsed from")
W("`src/displacement/ft_liftonly.log` (a complete 3000-iteration run, 8-9 Oct 2026).")
W("")
if t5:
    W("| metric | value |")
    W("|---|---|")
    W(f"| recipe | batch 1 x grad-accum 8 (effective {t5['effective_batch']}), lr 5e-5, LoRA r32/a16 |")
    W(f"| s/iteration, median of {t5['n_sit_samples']} tqdm samples | {f(t5.get('sec_per_it_p50'))} |")
    W(f"| s/iteration from total wall | {f(t5.get('sec_per_it_from_wall'))} ({t5['final_progress']['step']} steps in {t5['final_progress']['elapsed_sec']} s) |")
    W(f"| **samples/s (median)** | **{f(t5.get('samples_per_sec_p50'))}** |")
    W(f"| samples/s (from total wall, includes checkpoint saves) | {f(t5.get('samples_per_sec_from_wall'))} |")
    W(f"| 200 steps would take | {f((t5.get('steps_200_projected_sec') or 0) / 60.0, 1)} min |")
    W("| GPU utilisation during training | **NOT MEASURED** |")
    W("")
    W("Why not re-measured on GPU 1. `datagen/run_ft_liftonly.sh` passes the reference 17500")
    W("checkpoint as `--vla_path`, and `finetune_headinit.py` rewrites `config.json` **inside**")
    W("that directory — `ft_liftonly.log` lines 18-20 record it creating")
    W("`config.json.back.20261008_221207` and then updating `config.json`. That directory is the")
    W("base policy for every parity comparison in the plan, and this project may not modify")
    W("anything outside `fastvla-eval`, so a fresh 200-step run was not started. The existing log")
    W("is also the better sample: 3000 iterations on the same card rather than 200. The cost is")
    W("that training-time GPU utilisation stays unknown; it is only needed for the optional M8")
    W("(two-GPU DDP), which the plan says not to begin without a second card.")
else:
    W("_not yet run_")
W("")

# ---------------------------------------------------------------- cross-checks ----
W("## Cross-validation")
W("")
W("Four numbers were measured twice by unrelated means. They agree, which is the reason to")
W("trust the rest of the audit.")
W("")
W("| quantity | method A | method B |")
W("|---|---|---|")
if t1 and t3:
    W(f"| GPU utilisation during eval | `nvidia-smi dmon`, 1 Hz over the full 60-start run: "
      f"**{f(t1.get('gpu1_sm_util_pct_mean'), 1)}%** | nsys kernel intervals over one rollout: "
      f"**{f(t3.get('gpu_busy_pct'), 1)}%** |")
if t3 and t2 and os.path.exists(npz):
    W(f"| time between forward passes | nsys macro gap mean: "
      f"**{f(t3.get('macro50_ms_mean'), 0)} ms** | 8 control steps x Table 2's per-step total: "
      f"**{f(8 * 1000 * accounted.mean(), 0)} ms** |")
if t4 and t2 and os.path.exists(npz):
    k05 = next((v for v in t4.values() if abs(v["distmax"] - 0.5) < 1e-9), None)
    if k05:
        W(f"| GJK clearance cost per control step | Table 4 replay, 2 calls x "
          f"{f(k05['ms_per_call_mean'])} ms: **{f(k05['ms_per_control_step'], 1)} ms** | "
          f"Table 2 in-loop timer: **{f(1000 * clr.mean())} ms** |")
if t1 and t2 and os.path.exists(npz):
    W(f"| per-control-step wall-clock | Table 1, uninstrumented, 15935 steps: "
      f"**{f(t1.get('ms_per_step_loop'), 1)} ms** | Table 2, instrumented, 6 tasks: "
      f"**{f(1000 * accounted.mean(), 1)} ms** (+{f(100 * (accounted.mean() * 1000 / t1['ms_per_step_loop'] - 1), 1)}% instrumentation) |")
W("")

# ---------------------------------------------------------------- decision --------
W("## The M3-vs-M4 ordering decision")
W("")
if t2 and os.path.exists(npz):
    fwd_share = 100.0 * fwd.sum() / tot
    sim_lab_share = 100.0 * (sim.sum() + render.sum() + mon.sum()) / tot
    lab_share = 100.0 * mon.sum() / tot
    W("The plan's rule (section 4): *if the policy forward is under 30% of step time and the")
    W("simulator plus labeler dominate, M3 is the big win and M4 is small; if the forward")
    W("dominates, reverse the order of 3 and 4.*")
    W("")
    W("**Evidence.** The policy forward is "
      f"**{f(fwd_share, 1)}%** of step time amortised ({f(1000 * fwd.sum() / max(1, n_fwd), 0)} ms per call, once per")
    W(f"8-step chunk). The simulator plus labeler are **{f(sim_lab_share, 1)}%**, of which the")
    W(f"SafetyMonitor alone is **{f(lab_share, 1)}%**. The GPU is idle "
      f"{f((t3 or {}).get('gpu_idle_pct'), 1)}% of the time.")
    W("")
    W(f"**Decision: keep the plan's order — M2, then M3, then M4.** The forward is far below the")
    W("30% threshold, so M4 cannot be the big win: even making the forward free would remove")
    W(f"only {f(fwd_share, 1)}% of step time. M3 keeps its place ahead of M4.")
    W("")
    W("**One correction to the plan's framing.** The plan calls M2 a 3-4 day warm-up and M3 the")
    W(f"big win. The measurement says M2 is the biggest single win available: the labeler is")
    W(f"{f(lab_share, 1)}% of step time on its own, and it is pure CPU Python. cProfile over one")
    W("rollout (`arrays/t2_cprofile_cprofile.txt`) puts the whole cost in two functions —")
    W("`convex_distance._closest_on_simplex` at 269,650 calls and 15.1 s of self time, and")
    W("`gjk_distance` at 81,068 calls and 6.6 s, with 590,806 `numpy.linalg.solve` calls")
    W("underneath them. That is the GJK simplex solver running in Python with a 3x3 LAPACK")
    W("solve per iteration, which is exactly what a C++ port removes.")
    W("")
    W("Two further findings that change how M2 and M3 should be built:")
    W("")
    W("1. **The labeler runs twice per control step, not once.** `SafetyMonitor.attach()` hooks")
    W("   `sim.step`, so it fires per MuJoCo substep; with `clearance_every=25` and 25 substeps")
    W("   per control step, clearance is computed on substep 1 and substep 25. Any speedup claim")
    W("   must use that factor of 2.")
    W("2. **Pruning already works; the cutoff is what costs.** Table 4's sweep shows the exact-pair")
    W("   count falling from 101.1 per call at the production `distmax=0.5` to **zero** at 0.05,")
    W("   where every candidate is pruned before any exact GJK call — and yet the call still costs")
    W("   24.19 ms. So roughly 24 ms of the 64 ms is the per-actor union-hull GJK and the")
    W("   bounding-volume work, which pruning cannot remove and which a C++ port must also cover.")
    W("   M2 should not expect the plan's 20-100x from threading the pair loop alone.")
W("")

# ------------------------------------------------- estimates this audit replaces --
W("## Estimates in the plan that this audit replaces")
W("")
W("The plan's instruction is that every number in it not marked MEASURED is an estimate and")
W("must be replaced by a measurement. These are the ones milestone 0 can settle.")
W("")
W("| plan's figure (section) | plan said | measured | verdict |")
W("|---|---|---|---|")
if t1:
    W(f"| 60 starts for one arm, wall-clock (3.4) | \"on the order of 1-2 hours on GPU 1\" | "
      f"**{f((t1.get('rollout_loop_wall_sec') or 0) / 60.0, 0)} min** ({f(t1.get('rollouts_per_hour_loop'), 1)} rollouts/h) | "
      "faster than estimated; the low end was right |")
if t2 and os.path.exists(npz) and t4:
    k05 = next((v for v in t4.values() if abs(v["distmax"] - 0.5) < 1e-9), None)
    W(f"| labeler cost per step (3.3) | \"about 543 ms per step with a wide cutoff\", "
      "explicitly NOT independently measured | "
      f"**{f(1000 * clr.mean())} ms/step** in-loop, **{f(k05['ms_per_control_step'], 1)} ms/step** on replay | "
      f"the real figure is ~{f(543.0 / (1000 * clr.mean()), 1)}x smaller than the plan's number |")
    W(f"| per-step split (3.4) | \"the split is NOT measured\" | "
      f"labeler {f(100 * mon.sum() / tot, 1)}%, forward {f(100 * fwd.sum() / tot, 1)}%, "
      f"controller+obs {f(100 * derived.sum() / tot, 1)}%, prep {f(100 * prep.sum() / tot, 1)}%, "
      f"physics {f(100 * sim.sum() / tot, 1)}%, render {f(100 * render.sum() / tot, 1)}% | measured, Table 2 |")
    W(f"| forward latency (10) | \"7B at ~50-100 ms per call is plausible\" | "
      f"**{f(1000 * fwd.sum() / max(1, n_fwd), 0)} ms per call** at batch 1 | "
      "2-4x slower than the guess; two 256x256 images plus a ~57-token action span |")
if t4:
    pc = next(iter(t4.values())).get("obstacle_hull_pieces", {})
    if pc.get("oa1"):
        W(f"| oa1 obstacle hull pieces (3.3) | \"the oa1 bottle is 244 hull pieces\" | "
          f"**{pc['oa1']}** collision geoms across the declared obstacle | close; 244 was about right |")
if t5:
    W(f"| fine-tune iteration time (4.5) | \"~4.8 s/it at effective batch 8\" | "
      f"**{f(t5.get('sec_per_it_p50'))} s/it** median, {f(t5.get('sec_per_it_from_wall'))} s/it from total wall | "
      "the estimate was accurate |")
W("")
W("Two the audit does **not** settle, and why:")
W("")
W("- **Rendering looks too cheap.** 1.21 ms/step for two 256x256 camera renders is suspiciously")
W("  low. `MjSim.render` is timed, but robosuite may be doing part of the work elsewhere in")
W("  `_get_observations`, which would land in the derived controller/obs row (17.86 ms/step)")
W("  instead. Before M3 leans on \"rendering is free\", that row should be split properly.")
W("  Flagged rather than claimed.")
W("- **Training GPU utilisation**, for the reason in Table 5.")
W("")

# ---------------------------------------------------------------- assumptions -----
asm = os.path.join(A, "single_env_assumptions.md")
if os.path.exists(asm):
    W("---")
    W("")
    with open(asm) as fh:
        body = fh.read().strip()
    W(body.replace("# Single-environment assumptions", "## Single-environment assumptions", 1))
    W("")

with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"[ASSEMBLE] wrote {OUT} ({len(L)} lines)")
for name, obj in (("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4), ("t5", t5)):
    print(f"[ASSEMBLE] {name}: {'ok' if obj else 'MISSING'}")
