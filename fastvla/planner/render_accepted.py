"""Render the two accepted demos, with a check that the frames are actually images.

Three of the first eight clips produced in 7a were noise: the offscreen camera returned an
uninitialised buffer and the mp4 was written anyway. Pixels only -- clearance and success come
from physics and GJK, so no measured number was affected -- but a clip of static is worse than no
clip. This renders, VERIFIES, and retries.

The check: mean absolute difference between horizontally neighbouring pixels. A rendered LIBERO
scene sits around 2-10; uninitialised memory sits around 60-90. Anything above NOISE_THRESHOLD is
rejected and the scene is rebuilt.

Author: Varun Chandra.  Env: planner, GPU 1 only.  PICKS=shard:attempt,...
"""

import contextlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, CLIPS = os.path.join(HERE, "out"), os.path.join(HERE, "clips")
os.makedirs(CLIPS, exist_ok=True)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (
    DISP,
    os.path.join(DISP, "datagen"),
    "/data_new/vchandr2",
    "/data_new/vchandr2/fastvla-eval",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D  # noqa: E402
import geom_shift  # noqa: E402

from fastvla.planner import route as R  # noqa: E402
from fastvla.planner.step4b_run import draw_scene, via_clearances  # noqa: E402

PICKS = [
    tuple(int(x) for x in t.split(":")) for t in os.environ.get("PICKS", "5:9,8:10").split(",")
]
RES = int(os.environ.get("RES", "256"))
NOISE_THRESHOLD = float(os.environ.get("NOISE_THRESHOLD", "40"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))
BASE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")


def frame_score(fr):
    """Mean |horizontal neighbour difference|. Low = a real image, high = noise."""
    f = np.asarray(fr, dtype=np.int16)
    return float(np.abs(f[:, 1:] - f[:, :-1]).mean())


def write_video(path, frames, fps=20):
    import imageio.v2 as iio

    w = iio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )
    for fr in frames:
        w.append_data(np.ascontiguousarray(np.asarray(fr, np.uint8)))
    w.close()


def run_scene(k, a, base, obstacle, lo, hi):
    """Rebuild scene (k,a), execute the planned route with cameras on, return frames + metadata."""
    rng = np.random.RandomState(131 * k)
    for i in range(a + 1):
        init_idx = int(rng.randint(lo, hi + 1))
        env = None
        try:
            env, dom, mon, sg, task, init, obs = D.make_env("oa1", init_idx, res=RES)
            D.set_render(dom, True)
            # One warm render before anything is recorded. The first offscreen frame after a new
            # EGL context can come back as uninitialised memory; stepping once with the cameras
            # already enabled forces it to be filled.
            for _ in range(D.CONF["SETTLE_STEPS"]):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            cand, intr = draw_scene(sg, obstacle, base[f"oa1__i{init_idx:02d}__eef"], rng)
            if i < a or cand is None:
                continue
            shift = np.asarray(cand["shift"], float)
            _, _, anch = geom_shift.obstacle_anchors(dom)
            st = geom_shift.apply_shift(dom, anch, init[init_idx], shift)
            dom, mon, sg, obs, _ = D.setup_scene(env, st, init_idx, shift=shift, render=True)
            warm = frame_score(obs["agentview_image"])
            home = np.asarray(obs["robot0_eef_pos"], float)
            hq = np.asarray(obs["robot0_eef_quat"], float)
            wp0 = D.nominal_plan(sg, dom, "oa1", home, hq)
            R.reset_stats()
            wp, det, shape = D.plan_detours(
                sg, obstacle, wp0, ceiling=D.measure_ceiling("oa1")["usable_m"]
            )
            if not shape.get("plan_clears"):
                return None, dict(reason=f"no plan: {shape.get('reason')}")
            ep = D.Episode()
            ep.render = True
            ep.last_obs = obs
            mon.attach()
            t, ok, why = D.execute(env, dom, mon, wp, ep, t0=0, ref_quat=hq)
            mon.detach()
            clr_cm = float(np.min(ep.clearance)) if len(ep.clearance) else float("nan")
            c = via_clearances(dom, mon, wp, obstacle)
            frames = np.asarray(ep.base, np.uint8)
            return frames, dict(
                shard=k,
                attempt=a,
                init_idx=init_idx,
                shift_cm=round(100 * float(np.linalg.norm(shift)), 3),
                min_clearance_cm=round(clr_cm, 3),
                steps=int(t),
                executor_ok=bool(ok),
                executor_reason=why,
                success=bool(env.check_success()),
                planned_via_min_cm=round(100 * float(c.min()), 3) if len(c) else None,
                tilt_deg=(det or [{}])[0].get("tilt_deg"),
                warm_frame_score=round(warm, 2),
                frames=int(len(frames)),
            )
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
    return None, dict(reason="scene not reached")


def main():
    R.install("curobo")
    base = {
        k: np.asarray(v)
        for k, v in dict(np.load(BASE, allow_pickle=True)).items()
        if k.endswith("__eef")
    }
    obstacle = D.SPEC["names"]["oa1"]["obstacle"]
    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    for k, a in PICKS:
        for attempt in range(1, MAX_TRIES + 1):
            t0 = time.perf_counter()
            frames, meta = run_scene(k, a, base, obstacle, lo, hi)
            if frames is None or not len(frames):
                print(f"  s{k:02d}a{a:02d}: {meta.get('reason')}", flush=True)
                break
            scores = [frame_score(frames[i]) for i in (0, len(frames) // 2, len(frames) - 1)]
            worst = max(scores)
            if worst > NOISE_THRESHOLD:
                print(
                    f"  s{k:02d}a{a:02d}: try {attempt} produced NOISE "
                    f"(neighbour-diff {worst:.1f} > {NOISE_THRESHOLD}), retrying",
                    flush=True,
                )
                continue
            nm = (
                f"accepted_s{k:02d}a{a:02d}_i{meta['init_idx']:02d}_"
                f"shift{meta['shift_cm']:.1f}cm_clr{meta['min_clearance_cm']:.2f}cm"
            )
            write_video(os.path.join(CLIPS, nm + ".mp4"), frames)
            meta["frame_scores"] = [round(s, 2) for s in scores]
            meta["verified_real_frames"] = True
            with open(os.path.join(CLIPS, nm + ".json"), "w") as fh:
                json.dump(meta, fh, indent=1, default=str)
            print(
                f"  {nm}.mp4  {len(frames)} frames  neighbour-diff {worst:.1f} (real)  "
                f"clearance {meta['min_clearance_cm']} cm  success={meta['success']}  "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
            break
    print("RENDER_ACCEPTED_DONE")


if __name__ == "__main__":
    main()
