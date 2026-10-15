"""M3a: a plain PyTorch policy server with a request queue and dynamic batching.

One copy of the policy on GPU 1. Two threads, for the reason in DESIGN.md section 6:

  receiver thread  polls the ROUTER socket, runs prepare_request (~33 ms of CPU: crop,
                   normalise, tokenise) for each arrival, and pushes onto a queue
  forward thread   pops one request (blocking), then collects up to B-1 more that arrive
                   within D ms, runs ONE GPU forward, and replies to each caller

Preprocessing therefore overlaps the GPU forward, and the GIL is released inside the CUDA
call. Replies go back on the ROUTER socket from the forward thread via a reply queue owned
by the receiver thread, because a ZeroMQ socket must not be touched from two threads.

Env: RUN_ID, ENDPOINT, MAX_BATCH (B), BATCH_DELAY_MS (D), [PARITY=1, CKPT, STATS_OUT]
"""

import json
import os
import queue
import sys
import threading
import time

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    "/data_new/vchandr2/fastvla-eval",
):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402
import zmq  # noqa: E402

from fastvla.server import protocol as P  # noqa: E402

ENDPOINT = os.environ["ENDPOINT"]
RUN_ID = os.environ["RUN_ID"]
MAX_BATCH = int(os.environ.get("MAX_BATCH", "4"))
BATCH_DELAY_MS = float(os.environ.get("BATCH_DELAY_MS", "10"))
PARITY = os.environ.get("PARITY", "0") == "1"
STATS_OUT = os.environ.get("STATS_OUT", "")
BASE = (
    "openvla-7b-oft-libero-combined+libero_safety+b32+lr-0.0005+lora-r32+dropout-0.0"
    "--image_aug--libero_safety--warmstart-combined--headinit"
)
CKPT = os.environ.get("CKPT", f"/data_new/vchandr2/runs/{BASE}--17500_chkpt")

if PARITY and MAX_BATCH != 1:
    raise SystemExit("PARITY=1 requires MAX_BATCH=1 (DESIGN.md section 3)")
if os.environ.get("PROBE", "0") == "1" and not PARITY:
    raise SystemExit(
        "PROBE=1 is refused outside parity mode: the driver's probe hook indexes "
        "h[0, ...] (assumption B2 in audit/single_env_assumptions.md), so at batch > 1 it "
        "would silently return batch element 0 for every request. See DESIGN.md section 5."
    )

STATS = dict(
    forwards=0,
    requests=0,
    batch_hist={},
    forward_ms=[],
    queue_wait_ms=[],
    prep_ms=[],
    max_batch=MAX_BATCH,
    batch_delay_ms=BATCH_DELAY_MS,
    parity=PARITY,
    endpoint=ENDPOINT,
    run_id=RUN_ID,
)


def build_policy():
    import torch
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_model
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    from fastvla.server.batched_forward import BatchedPolicy

    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=CKPT,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        lora_rank=32,
        env_img_res=256,
    )
    model = get_model(cfg)
    pp = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    ah = get_action_head(cfg, model.llm_dim)
    proc = get_processor(cfg)
    cfg.unnorm_key = "libero_safety"
    pol = BatchedPolicy(cfg, model, proc, ah, pp)
    print(
        f"[SERVER] policy loaded, torch {torch.__version__}, "
        f"B<={MAX_BATCH}, D={BATCH_DELAY_MS} ms, parity={PARITY}",
        flush=True,
    )
    return pol


def main():
    pol = build_policy()
    ctx = zmq.Context(io_threads=2)
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(ENDPOINT)
    work = queue.Queue()
    replies = queue.Queue()
    state = dict(workers=0, hellos=0, byes=0, stop=False)
    print(f"[SERVER] READY {ENDPOINT}", flush=True)

    def receiver():
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not state["stop"]:
            # drain replies first so callers are never left waiting behind a recv
            try:
                while True:
                    ident, frames = replies.get_nowait()
                    sock.send_multipart([ident] + frames)
            except queue.Empty:
                pass
            if not dict(poller.poll(timeout=5)):
                continue
            ident, *frames = sock.recv_multipart()
            kind = frames[0]
            if kind == P.REQ:
                t0 = time.perf_counter()
                header, full, wrist, st = P.unpack_request(frames)
                prepared = pol.prepare_request(full, wrist, st, header["instruction"])
                prep_ms = 1000 * (time.perf_counter() - t0)
                STATS["prep_ms"].append(prep_ms)
                work.put((ident, header, prepared, time.perf_counter()))
            elif kind == P.HELLO:
                state["hellos"] += 1
                state["workers"] += 1
                print(f"[SERVER] worker joined ({state['workers']} active)", flush=True)
            elif kind == P.BYE:
                state["byes"] += 1
                state["workers"] -= 1
                print(f"[SERVER] worker left ({state['workers']} active)", flush=True)
                if state["workers"] <= 0 and state["hellos"] > 0:
                    state["stop"] = True

    def forward_loop():
        delay = BATCH_DELAY_MS / 1000.0
        while not state["stop"]:
            try:
                first = work.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            # Only requests with the SAME prompt length may share a forward: the six task
            # instructions tokenise to different lengths, and padding to match them shifts
            # rotary positions and corrupts the actions (see
            # batched_forward.assert_uniform_prompt_length). Non-matching arrivals are set
            # aside and re-queued, so they are not dropped or reordered relative to each
            # other -- they just wait for a batch of their own shape.
            want_len = int(first[2]["input_ids"].shape[1])
            deferred = []
            if MAX_BATCH > 1 and delay > 0:
                deadline = time.perf_counter() + delay
                while len(batch) < MAX_BATCH:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = work.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if int(item[2]["input_ids"].shape[1]) == want_len:
                        batch.append(item)
                    else:
                        deferred.append(item)
            elif MAX_BATCH > 1:
                while len(batch) < MAX_BATCH:
                    try:
                        item = work.get_nowait()
                    except queue.Empty:
                        break
                    if int(item[2]["input_ids"].shape[1]) == want_len:
                        batch.append(item)
                    else:
                        deferred.append(item)
            for item in deferred:
                work.put(item)
            STATS["deferred_for_length"] = STATS.get("deferred_for_length", 0) + len(deferred)
            now = time.perf_counter()
            waits = [1000 * (now - b[3]) for b in batch]
            t0 = time.perf_counter()
            chunks = pol.forward([b[2] for b in batch])
            fms = 1000 * (time.perf_counter() - t0)
            STATS["forwards"] += 1
            STATS["requests"] += len(batch)
            STATS["forward_ms"].append(fms)
            STATS["queue_wait_ms"].extend(waits)
            k = str(len(batch))
            STATS["batch_hist"][k] = STATS["batch_hist"].get(k, 0) + 1
            for (ident, header, _prep, _t), chunk, w in zip(batch, chunks, waits, strict=True):
                h = dict(header)
                h.update(server_batch_size=len(batch), queue_wait_ms=w, forward_ms=fms)
                replies.put((ident, P.pack_response(h, chunk)))

    rt = threading.Thread(target=receiver, daemon=True)
    ft = threading.Thread(target=forward_loop, daemon=True)
    rt.start()
    ft.start()
    try:
        while not state["stop"]:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    state["stop"] = True
    time.sleep(0.3)

    if STATS["forward_ms"]:
        f = np.array(STATS["forward_ms"])
        q_ = np.array(STATS["queue_wait_ms"])
        pm = np.array(STATS["prep_ms"]) if STATS["prep_ms"] else np.zeros(1)
        STATS["summary"] = dict(
            forwards=STATS["forwards"],
            requests=STATS["requests"],
            mean_batch=STATS["requests"] / max(1, STATS["forwards"]),
            forward_ms_mean=float(f.mean()),
            forward_ms_p95=float(np.percentile(f, 95)),
            queue_wait_ms_mean=float(q_.mean()),
            queue_wait_ms_p95=float(np.percentile(q_, 95)),
            prep_ms_mean=float(pm.mean()),
            requests_per_s_gpu=1000.0 * STATS["requests"] / max(1e-9, f.sum()),
        )
        print(
            f"[SERVER] {STATS['forwards']} forwards, {STATS['requests']} requests, "
            f"mean batch {STATS['summary']['mean_batch']:.2f}, "
            f"forward {STATS['summary']['forward_ms_mean']:.1f} ms, "
            f"queue wait {STATS['summary']['queue_wait_ms_mean']:.1f} ms, "
            f"prep {STATS['summary']['prep_ms_mean']:.1f} ms",
            flush=True,
        )
    for k in ("forward_ms", "queue_wait_ms", "prep_ms"):
        STATS[k] = STATS[k][:5000]
    if STATS_OUT:
        with open(STATS_OUT, "w") as fh:
            json.dump(STATS, fh, indent=2)
    print("[SERVER] DONE", flush=True)


if __name__ == "__main__":
    main()
