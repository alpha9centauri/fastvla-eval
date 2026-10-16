"""Manifest runner: start the server, spawn N workers, merge per-rollout results.

Workers are separate OS processes started with subprocess, not multiprocessing: each needs
its own EGL context and its own `rollout_bootstrap` import order, and forking a process
that has touched CUDA or EGL is a known way to get silent corruption. One process per
environment is also what DESIGN.md specifies.

This process owns the single `[PROGRESS] done=N total=N` line that run_with_alert.py
parses (assumption A5); each worker's stdout goes to its own log file.

Merging is a pure function of `rollouts/`: every worker writes one file per rollout, so
the merge is a directory listing, and a worker that dies loses at most its current
rollout. Re-running the same run_id resumes, because a worker skips a rollout whose file
already exists.

Env: RUN_ROOT, MANIFEST (ctrl60|gate10), N_WORKERS, MAX_BATCH, BATCH_DELAY_MS,
     [RUN_ID, PARITY=1, LABELER=cpp, CONTACTS=1, MAX_STEPS=300, VIDEO_KEYS]
"""

import json
import os
import subprocess
import sys
import time
import uuid

# Derived, not hardcoded, so a fresh clone runs against itself rather than
# against whatever repo happens to live at an absolute path.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = "/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python"

RUN_ROOT = os.environ.get("RUN_ROOT", os.path.join(ROOT, "bench", "runs"))
MANIFEST_NAME = os.environ.get("MANIFEST", "gate10")
N_WORKERS = int(os.environ.get("N_WORKERS", "1"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "1"))
BATCH_DELAY_MS = float(os.environ.get("BATCH_DELAY_MS", "10"))
PARITY = os.environ.get("PARITY", "0") == "1"
RUN_ID = os.environ.get("RUN_ID") or (
    f"{MANIFEST_NAME}_n{N_WORKERS}_b{MAX_BATCH}_d{int(BATCH_DELAY_MS)}"
    f"_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
)


def main():
    sys.path.insert(0, ROOT)
    from fastvla.eval import manifest as M

    specs = M.build(MANIFEST_NAME)
    if PARITY and (N_WORKERS != 1 or MAX_BATCH != 1):
        raise SystemExit("PARITY=1 requires N_WORKERS=1 and MAX_BATCH=1")

    run_dir = os.path.join(RUN_ROOT, RUN_ID)
    if os.path.exists(os.path.join(run_dir, "merged.json")):
        raise SystemExit(f"run {RUN_ID} already complete at {run_dir}")
    for d in ("rollouts", "arrays", "videos", "logs"):
        os.makedirs(os.path.join(run_dir, d), exist_ok=True)
    # POLICY_TRANSPORT=triton drives an already-running Triton server (started separately
    # by bench/triton_serve.sh inside the container) instead of spawning the ZeroMQ one,
    # so the comparison is of serving layers with the same workers and the same manifest.
    transport = os.environ.get("POLICY_TRANSPORT", "zmq")
    if transport == "triton":
        endpoint = os.environ.get("TRITON_URL", "localhost:8201")
    else:
        endpoint = f"ipc:///tmp/fastvla_{RUN_ID}.sock"

    # round-robin so every worker gets a mix of tasks rather than one worker taking all
    # of the slow ones; the manifest is ordered task-major.
    slices = [specs[i::N_WORKERS] for i in range(N_WORKERS)]
    cfg = dict(
        run_id=RUN_ID,
        manifest=MANIFEST_NAME,
        n_rollouts=len(specs),
        n_workers=N_WORKERS,
        max_batch=MAX_BATCH,
        batch_delay_ms=BATCH_DELAY_MS,
        parity=PARITY,
        endpoint=endpoint,
        labeler=os.environ.get("LABELER", "cpp"),
        cudnn_benchmark=os.environ.get("CUDNN_BENCHMARK", "0"),
        torch_deterministic=os.environ.get("TORCH_DETERMINISTIC", "0"),
        server_opts=os.environ.get("SERVER_OPTS", ""),
        capture_activations=os.environ.get("CAPTURE_ACTIVATIONS", "0"),
        ckpt=os.environ.get("CKPT", ""),
        monitor=os.environ.get("MONITOR", "off"),
        monitor_probe=os.environ.get("MONITOR_PROBE", ""),
        monitor_threshold=os.environ.get("MONITOR_THRESHOLD", ""),
        monitor_debounce=os.environ.get("MONITOR_DEBOUNCE", ""),
        contacts=os.environ.get("CONTACTS", "1"),
        per_worker=[len(s) for s in slices],
    )
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(
        f"[RUN] {RUN_ID}: {len(specs)} rollouts, N={N_WORKERS}, B={MAX_BATCH}, "
        f"D={BATCH_DELAY_MS} ms, parity={PARITY}",
        flush=True,
    )

    env_common = dict(os.environ)
    env_common.update(
        CUDA_VISIBLE_DEVICES="1",
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="1",
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PYTHONPATH="/data_new/vchandr2/rollout_stubs",
        LABELER=os.environ.get("LABELER", "cpp"),
        CONTACTS=os.environ.get("CONTACTS", "1"),
        LABELER_LEVEL=os.environ.get("LABELER_LEVEL", "batch"),
        LABELER_THREADS=os.environ.get("LABELER_THREADS", "1"),
    )

    server_env = dict(env_common)
    server_env.update(
        ENDPOINT=endpoint,
        RUN_ID=RUN_ID,
        MAX_BATCH=str(MAX_BATCH),
        BATCH_DELAY_MS=str(BATCH_DELAY_MS),
        PARITY="1" if PARITY else "0",
        CUDNN_BENCHMARK=os.environ.get("CUDNN_BENCHMARK", "0"),
        TORCH_DETERMINISTIC=os.environ.get("TORCH_DETERMINISTIC", "0"),
        SERVER_OPTS=os.environ.get("SERVER_OPTS", ""),
        CAPTURE_ACTIVATIONS=os.environ.get("CAPTURE_ACTIVATIONS", "0"),
        CKPT=os.environ.get("CKPT", ""),
        MONITOR=os.environ.get("MONITOR", "off"),
        MONITOR_PROBE=os.environ.get("MONITOR_PROBE", ""),
        MONITOR_THRESHOLD=os.environ.get("MONITOR_THRESHOLD", ""),
        MONITOR_DEBOUNCE=os.environ.get("MONITOR_DEBOUNCE", ""),
        STATS_OUT=os.path.join(run_dir, "server_stats.json"),
    )
    # these handles must outlive the block: they are the subprocesses' stdout for the
    # life of the run, so a context manager is the wrong shape here.
    slog = open(os.path.join(run_dir, "logs", "server.log"), "w", buffering=1)  # noqa: SIM115
    t_start = time.perf_counter()
    server = subprocess.Popen(
        [PY, "-u", os.path.join(ROOT, "fastvla/server/server.py")],
        env=server_env,
        stdout=slog,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    # wait for READY
    if transport == "triton":
        import tritonclient.grpc as _g

        cli = _g.InferenceServerClient(url=endpoint)
        deadline = time.time() + 600
        ready = False
        while time.time() < deadline:
            try:
                if cli.is_server_live() and cli.is_model_ready("openvla_oft"):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        if not ready:
            raise SystemExit(f"triton at {endpoint} not ready within 600 s")
        cli.close()
        print(f"[RUN] triton ready at {endpoint}", flush=True)
    ready, deadline = (transport == "triton"), time.time() + 600
    while transport == "zmq" and time.time() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"server exited early with {server.returncode}; see logs/server.log")
        with open(os.path.join(run_dir, "logs", "server.log")) as fh:
            if "[SERVER] READY" in fh.read():
                ready = True
                break
        time.sleep(0.5)
    if not ready:
        if server is not None:
            server.kill()
        raise SystemExit("server did not become ready within 600 s")
    t_ready = time.perf_counter()
    print(f"[RUN] server ready in {t_ready - t_start:.1f} s", flush=True)

    procs = []
    for wid, sl in enumerate(slices):
        wenv = dict(env_common)
        wenv.update(
            RUN_DIR=run_dir,
            ENDPOINT=endpoint,
            WORKER_ID=str(wid),
            RUN_ID=RUN_ID,
            MANIFEST_JSON=json.dumps(sl),
            MAX_STEPS=os.environ.get("MAX_STEPS", "300"),
            VIDEO_KEYS=os.environ.get("VIDEO_KEYS", ""),
        )
        wlog = open(os.path.join(run_dir, "logs", f"worker{wid}.log"), "w", buffering=1)  # noqa: SIM115
        procs.append(
            (
                wid,
                subprocess.Popen(
                    [PY, "-u", os.path.join(ROOT, "fastvla/sim/worker.py")],
                    env=wenv,
                    stdout=wlog,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                ),
                wlog,
            )
        )
    print(f"[RUN] spawned {len(procs)} workers", flush=True)

    rollouts_dir = os.path.join(run_dir, "rollouts")
    done_seen, total = 0, len(specs)
    while True:
        alive = [p for _w, p, _l in procs if p.poll() is None]
        n_done = len([f for f in os.listdir(rollouts_dir) if f.endswith(".json")])
        if n_done != done_seen:
            done_seen = n_done
            print(f"[PROGRESS] done={n_done} total={total}", flush=True)
        if not alive:
            break
        time.sleep(1.0)
    t_workers = time.perf_counter()

    rcs = {}
    for wid, p, lg in procs:
        rcs[wid] = p.wait()
        lg.close()
    if server is not None:
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
    if slog is not None:
        slog.close()

    rows = []
    for f in sorted(os.listdir(rollouts_dir)):
        if f.endswith(".json"):
            with open(os.path.join(rollouts_dir, f)) as fh:
                rows.append(json.load(fh))
    wall = t_workers - t_ready
    merged = dict(
        config=cfg,
        worker_returncodes=rcs,
        server_returncode=(server.returncode if server is not None else None),
        n_rows=len(rows),
        n_expected=total,
        wall_sec_rollouts=wall,
        wall_sec_incl_server_start=t_workers - t_start,
        rollouts_per_hour=(3600.0 * len(rows) / wall) if wall > 0 else None,
        total_steps=int(sum(r["steps"] for r in rows)),
        steps_per_sec=(sum(r["steps"] for r in rows) / wall) if wall > 0 else None,
        collisions=int(sum(bool(r["collision"]) for r in rows)),
        completions=int(sum(bool(r["completion"]) for r in rows)),
        gate_collisions=int(sum(bool(r.get("gate_collision")) for r in rows)),
        rows=rows,
    )
    sp = os.path.join(run_dir, "server_stats.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            merged["server"] = json.load(fh).get("summary")
    with open(os.path.join(run_dir, "merged.json"), "w") as fh:
        json.dump(merged, fh, indent=2)
    print(
        f"[RUN] merged {len(rows)}/{total} rollouts in {wall:.1f} s "
        f"= {merged['rollouts_per_hour']:.1f} rollouts/hour; "
        f"collisions {merged['collisions']}, completions {merged['completions']}",
        flush=True,
    )
    print(f"[RUN] -> {os.path.join(run_dir, 'merged.json')}", flush=True)
    if len(rows) != total or any(v != 0 for v in rcs.values()):
        print(f"[RUN] INCOMPLETE: {len(rows)}/{total} rows, worker rcs {rcs}", flush=True)
        return 3
    print("RUNNER DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
