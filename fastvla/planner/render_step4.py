"""Render three step-4 demos to fastvla/planner/clips/.

Nothing was accepted, and more than that: no scene produced an episode at all, so there are no
frames to pull out of the measurement run. `generate_demo` only renders the replay of a demo that
has already certified. So the three closest-to-accepted scenes are rebuilt here and the planner's
route is executed with the cameras on, through dgcore's own executor and monitor, and recorded
whatever the outcome. The clips therefore show what the planned route actually does -- which is
the thing worth looking at when the planner succeeds and the pipeline still produces nothing.

Ranking, from step4_rows.json: certification (the executor ran the route and the GJK clearance
reached 0.00) ranks above executor (a waypoint was never reached) above plan_fail above
no_clear_handoff. Within the top tier the cheapest to reproduce are chosen, because reproducing
attempt N means replaying that shard's rng through N prior scenes.

Scene identity is checked, not assumed: the rebuilt init index and shift must match the recorded
ones or the render is refused.

Author: Varun Chandra
Env: planner. GPU 1 only.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CLIPS = os.path.join(HERE, "clips")
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

TAG = "oa1"
SEED = int(os.environ.get("SEED", "0"))
RES = int(os.environ.get("RES", "256"))
PICKS = os.environ.get("PICKS", "9:0,2:6,5:7")  # shard:attempt, chosen from the report
BASELINE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")


def write_video(path, frames, fps=20):
    """H.264 via imageio-ffmpeg.

    dgcore's own writer (dglerobot.write_video) uses PyAV, and PyAV is not in the planner env
    while pandas -- which dglerobot imports at module level -- is not either. Rather than add two
    packages to an environment that is already carrying cuRobo and LIBERO together, the clips are
    written here with imageio, which is present. Same codec, same frame data.
    """
    import imageio.v2 as iio

    w = iio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )
    for fr in frames:
        w.append_data(np.ascontiguousarray(np.asarray(fr, np.uint8)))
    w.close()


def draw_shift(sg, obstacle, traj, rng):
    """generate_demo's own eval-rule draw loop, replicated so the rng advances identically."""
    for _ in range(max(1, 2 * D.CONF["SCENE_TRIES_PER_SIGN"])):
        i_ = float(rng.uniform(D.CONF["EVAL_RULE_INTR_MIN_M"], D.CONF["EVAL_RULE_INTR_MAX_M"]))
        cand = D.eval_rule_shift(sg, obstacle, traj, i_)
        if cand.get("feasible"):
            return cand, i_
    return None, None


def main():
    with open(os.path.join(OUT, "step4_rows.json")) as fh:
        rows = {(r["shard"], r["attempt"]): r for r in json.load(fh)}
    picks = []
    for tok in PICKS.split(","):
        k, a = (int(x) for x in tok.split(":"))
        if (k, a) in rows:
            picks.append(rows[(k, a)])
    print(
        f"rendering {len(picks)} scenes (0 accepted in 200, so these are closest-to-accepted: "
        f"{', '.join(p['cause'] for p in picks)})",
        flush=True,
    )

    R.install("curobo")
    base = {
        kk: np.asarray(v)
        for kk, v in dict(np.load(BASELINE, allow_pickle=True)).items()
        if kk.endswith("__eef")
    }
    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    obstacle = D.SPEC["names"][TAG]["obstacle"]

    for r in picks:
        k, a = r["shard"], r["attempt"]
        t0 = time.perf_counter()
        rng = np.random.RandomState(SEED + 7919 * 0 + 131 * k)
        # replay the rng through the prior attempts of this shard
        for _i in range(a):
            ii = int(rng.randint(lo, hi + 1))
            D.generate_demo(TAG, ii, "varied", rng, res=84, placement="eval_rule", baseline=base)
        init_idx = int(rng.randint(lo, hi + 1))
        if init_idx != r["init_idx"]:
            print(
                f"  s{k:02d}a{a:02d}: init {init_idx} != recorded {r['init_idx']}, refusing",
                flush=True,
            )
            continue

        env = None
        try:
            env, dom, mon, sg, task, init, obs = D.make_env(TAG, init_idx, res=RES)
            D.set_render(dom, True)
            for _ in range(D.CONF["SETTLE_STEPS"]):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            cand, intr = draw_shift(sg, obstacle, base[f"{TAG}__i{init_idx:02d}__eef"], rng)
            if cand is None:
                print(f"  s{k:02d}a{a:02d}: no feasible placement on rebuild, refusing", flush=True)
                continue
            shift = np.asarray(cand["shift"], float)
            shift_cm = round(100 * float(np.linalg.norm(shift)), 3)
            if r["shift_cm"] is not None and abs(shift_cm - r["shift_cm"]) > 0.02:
                print(
                    f"  s{k:02d}a{a:02d}: shift {shift_cm} != recorded {r['shift_cm']}, refusing",
                    flush=True,
                )
                continue
            _, _, anch = geom_shift.obstacle_anchors(dom)
            st = geom_shift.apply_shift(dom, anch, init[init_idx], shift)
            dom, mon, sg, obs, _ = D.setup_scene(env, st, init_idx, shift=shift, render=True)

            home = np.asarray(obs["robot0_eef_pos"], float).copy()
            hq = np.asarray(obs["robot0_eef_quat"], float).copy()
            wp0 = D.nominal_plan(sg, dom, TAG, home, hq)
            R.reset_stats()
            wp, detours, shape = D.plan_detours(
                sg, obstacle, wp0, ceiling=D.measure_ceiling(TAG)["usable_m"]
            )
            det = (detours or [{}])[0] if detours else {}
            if not shape.get("plan_clears"):
                print(
                    f"  s{k:02d}a{a:02d}: route stage gave no path on rebuild "
                    f"({shape.get('reason')}), refusing",
                    flush=True,
                )
                continue

            ep = D.Episode()
            ep.render = True
            ep.last_obs = obs
            mon.attach()
            t, ok, why = D.execute(env, dom, mon, wp, ep, t0=0, ref_quat=hq)
            mon.detach()
            clr = float(np.min(ep.clearance)) if len(ep.clearance) else float("nan")
            frames = np.asarray(ep.base, np.uint8)
            nm = f"step4_s{k:02d}a{a:02d}_i{init_idx:02d}_shift{shift_cm:.0f}cm_{r['cause']}"
            if len(frames):
                write_video(os.path.join(CLIPS, nm + ".mp4"), frames)
            meta = dict(
                shard=k,
                attempt=a,
                init_idx=init_idx,
                shift_cm=shift_cm,
                intrusion_cm=round(100 * intr, 2),
                cause_in_measurement=r["cause"],
                executor_ok=bool(ok),
                executor_reason=why,
                steps=int(t),
                min_gjk_clearance_cm=round(100 * clr, 3),
                handoff_up_cm=r["handoff_up_cm"],
                handoff_out_cm=r["handoff_out_cm"],
                tilt_deg=det.get("tilt_deg"),
                n_via=det.get("n_via"),
                payload_attached=det.get("payload_attached"),
                frames=int(len(frames)),
                rebuild_seconds=round(time.perf_counter() - t0, 1),
            )
            with open(os.path.join(CLIPS, nm + ".json"), "w") as fh:
                json.dump(meta, fh, indent=1, default=str)
            print(
                f"  {nm}.mp4  {len(frames)} frames  executor_ok={ok} ({why})  "
                f"steps {t}  min GJK clearance {100 * clr:.2f} cm  "
                f"tilt {det.get('tilt_deg')}  ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
    print("RENDER_STEP4_DONE")


if __name__ == "__main__":
    main()
