"""Table 2: per-step wall-clock breakdown of the existing driver.

Runs the unmodified `rollout_openvla_disp.py` under runpy with profile_hooks installed.
Three modes, because the instruments interfere with each other:

  MODE=timers    manual perf_counter timers only. This is the mode the breakdown
                 numbers come from -- nothing else is attached, so the wall-clock is
                 the driver's own.
  MODE=torch     torch.profiler over a window of policy forwards, for the GPU-side
                 op split. CUDA activity tracing inflates CPU time, so the timers
                 collected in this mode are NOT used for the table.
  MODE=cprofile  cProfile over the whole run, saved as a .prof. Python-level call
                 counts; also inflates wall-clock.

Env: MODE, OUT (output path stem), plus every variable the driver itself reads.
"""
import json
import os
import runpy
import sys
import time

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in ("/data_new/vchandr2", "/data_new/vchandr2/openvla-oft", DISP):
    sys.path.insert(0, p)

import rollout_bootstrap  # noqa: F401  (must precede openvla_utils; keeps TF out of inference)

import profile_hooks

MODE = os.environ.get("MODE", "timers")
OUT = os.environ["OUT"]
DRIVER = os.path.join(DISP, "rollout_openvla_disp.py")

COUNT_PAIRS = os.environ.get("COUNT_PAIRS", "1") == "1"
patched = profile_hooks.install(count_pairs=COUNT_PAIRS)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def gpu_meta():
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        q = f"(nvidia-smi failed: {e})"
    return q


META = dict(
    mode=MODE,
    count_pairs=COUNT_PAIRS,
    driver=DRIVER,
    driver_md5=__import__("hashlib").md5(open(DRIVER, "rb").read()).hexdigest(),
    patched=patched,
    torch=torch.__version__,
    numpy=np.__version__,
    cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda"),
    nvidia_smi=gpu_meta(),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    mujoco_egl_device_id=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    env={k: os.environ.get(k) for k in
         ("OUTDIR", "STEP", "N_INIT", "SMOKE_INITS", "MAX_STEPS", "PROBE", "OFFSETS",
          "ARMS", "USE_FEASIBLE", "WRITE_BASELINE", "STEER_K", "INSTR_MODE")},
    started=time.strftime("%Y-%m-%d %H:%M:%S"),
)

t0 = time.perf_counter()

if MODE == "torch":
    from torch.profiler import ProfilerActivity, profile, schedule
    # wait 4 forwards (model warm-up and the first rollout's transients), warm up 2,
    # then record 12. prof.step() is called from the get_action wrapper.
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=4, warmup=2, active=12, repeat=1),
        record_shapes=True, profile_memory=False, with_stack=False,
    )
    prof.start()
    profile_hooks.set_profiler(prof)
    try:
        runpy.run_path(DRIVER, run_name="__main__")
    finally:
        prof.stop()
        ka = prof.key_averages()
        with open(OUT + "_torchprof_cuda.txt", "w") as f:
            f.write(ka.table(sort_by="self_cuda_time_total", row_limit=60))
        with open(OUT + "_torchprof_cpu.txt", "w") as f:
            f.write(ka.table(sort_by="self_cpu_time_total", row_limit=60))
        try:
            prof.export_chrome_trace(OUT + "_trace.json")
        except Exception as e:
            print(f"[WARN] chrome trace export failed: {e}", flush=True)
        tot_cuda = sum(e.self_cuda_time_total for e in ka)
        tot_cpu = sum(e.self_cpu_time_total for e in ka)
        json.dump(dict(self_cuda_us_total=tot_cuda, self_cpu_us_total=tot_cpu,
                       n_events=len(ka)), open(OUT + "_torchprof_totals.json", "w"), indent=2)
elif MODE == "cprofile":
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    try:
        runpy.run_path(DRIVER, run_name="__main__")
    finally:
        pr.disable()
        pr.dump_stats(OUT + ".prof")
        with open(OUT + "_cprofile.txt", "w") as f:
            st = pstats.Stats(pr, stream=f)
            st.sort_stats("cumulative").print_stats(60)
            st.sort_stats("tottime").print_stats(60)
else:
    runpy.run_path(DRIVER, run_name="__main__")

META["wall_sec_total"] = time.perf_counter() - t0
keys, arr = profile_hooks.save(OUT, META)

# ---- summary to stdout (the log is the only place console output survives) --------
def pct(x, tot):
    return f"{100.0 * x / tot:5.1f}%" if tot > 0 else "    - "


if len(arr):
    n = len(arr)
    tot_by = dict(zip(keys, arr.sum(axis=0)))
    # env_step already contains sim + monitor + render; derive what is left in it.
    inside = tot_by.get("sim", 0) + tot_by.get("monitor", 0) + tot_by.get("render", 0)
    derived_ctrl_obs = tot_by.get("env_step", 0) - inside
    accounted = (tot_by.get("env_step", 0) + tot_by.get("prep", 0)
                 + tot_by.get("forward", 0) + tot_by.get("proc", 0))
    print(f"\n[T2] control steps measured: {n}")
    print(f"[T2] accounted per-step total: {accounted:.1f} s "
          f"({1000 * accounted / n:.1f} ms/step)")
    rows = [("MuJoCo physics (sim.step)", tot_by.get("sim", 0)),
            ("rendering (2 cameras)", tot_by.get("render", 0)),
            ("SafetyMonitor total", tot_by.get("monitor", 0)),
            ("  of which GJK clearance", tot_by.get("clearance", 0)),
            ("OSC controller + obs assembly (derived)", derived_ctrl_obs),
            ("obs preprocessing (prepare_observation)", tot_by.get("prep", 0)),
            ("policy forward (get_action)", tot_by.get("forward", 0)),
            ("process_action", tot_by.get("proc", 0))]
    print(f"{'component':42s} {'total s':>9s} {'ms/step':>9s} {'share':>7s}")
    for name, v in rows:
        print(f"{name:42s} {v:9.2f} {1000 * v / n:9.2f} "
              f"{pct(v, accounted) if not name.startswith('  ') else '       '}")
    print(f"[T2] calls: {json.dumps(dict(profile_hooks.CALLS))}")
    if profile_hooks.PAIRS:
        P = np.array(profile_hooks.PAIRS)
        print(f"[T2] hull pairs per clearance call: mean {P.mean():.1f} "
              f"p95 {np.percentile(P, 95):.0f} max {P.max()} over {len(P)} calls")
print(f"[T2] arrays -> {OUT}.npz   meta -> {OUT}.json")
print("T2 BREAKDOWN DONE", flush=True)
