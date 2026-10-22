"""Render three step-4b scenes to clips/. Nothing accepted, so these are closest-to-accepted:
the three with the best planned-via clearance among the 55 that planned, chosen cheap-first.

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
import dgcore as D  # noqa: E402  sys.path
import geom_shift  # noqa: E402  sys.path

from fastvla.planner import route as R  # noqa: E402  sys.path
from fastvla.planner import world as W  # noqa: E402  sys.path
from fastvla.planner.step4b_run import (  # noqa: E402  sys.path
    arrival_errors,
    draw_scene,
    via_clearances,
)

PICKS = [
    tuple(int(x) for x in t.split(":")) for t in os.environ.get("PICKS", "8:3,6:3,2:9").split(",")
]
RES = int(os.environ.get("RES", "256"))
BASE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")


def write_video(path, frames, fps=20):
    import imageio.v2 as iio

    w = iio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1, ffmpeg_log_level="error"
    )
    for fr in frames:
        w.append_data(np.ascontiguousarray(np.asarray(fr, np.uint8)))
    w.close()


def main():
    R.install("curobo")
    with open(os.path.join(OUT, "step4b_full.json")) as fh:
        rows = {(r["shard"], r["attempt"]): r for r in json.load(fh)}
    base = {
        k: np.asarray(v)
        for k, v in dict(np.load(BASE, allow_pickle=True)).items()
        if k.endswith("__eef")
    }
    obstacle = D.SPEC["names"]["oa1"]["obstacle"]
    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    print(f"rendering {len(PICKS)} closest-to-accepted scenes (0 of 129 accepted)", flush=True)
    for k, a in PICKS:
        rec = rows.get((k, a), {})
        rng = np.random.RandomState(131 * k)
        t0 = time.perf_counter()
        for i in range(a + 1):
            init_idx = int(rng.randint(lo, hi + 1))
            env = None
            try:
                env, dom, mon, sg, task, init, obs = D.make_env(
                    "oa1", init_idx, res=RES if i == a else 84
                )
                D.set_render(dom, i == a)
                for _ in range(D.CONF["SETTLE_STEPS"]):
                    obs, *_ = env.step([0.0] * 6 + [-1.0])
                cand, intr = draw_scene(sg, obstacle, base[f"oa1__i{init_idx:02d}__eef"], rng)
                if i < a or cand is None:
                    continue
                shift = np.asarray(cand["shift"], float)
                _, _, anch = geom_shift.obstacle_anchors(dom)
                st = geom_shift.apply_shift(dom, anch, init[init_idx], shift)
                dom, mon, sg, obs, _ = D.setup_scene(env, st, init_idx, shift=shift, render=True)
                home = np.asarray(obs["robot0_eef_pos"], float)
                hq = np.asarray(obs["robot0_eef_quat"], float)
                wp0 = D.nominal_plan(sg, dom, "oa1", home, hq)
                R.reset_stats()
                wp, det, shape = D.plan_detours(
                    sg, obstacle, wp0, ceiling=D.measure_ceiling("oa1")["usable_m"]
                )
                if not shape.get("plan_clears"):
                    print(
                        f"  s{k:02d}a{a:02d}: no plan on rebuild ({shape.get('reason')})",
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
                c = via_clearances(dom, mon, wp, obstacle)
                d0 = (det or [{}])[0]
                nm = (
                    f"step4b_s{k:02d}a{a:02d}_i{init_idx:02d}_"
                    f"shift{100 * np.linalg.norm(shift):.0f}cm_executor"
                )
                write_video(os.path.join(CLIPS, nm + ".mp4"), np.asarray(ep.base, np.uint8))
                meta = dict(
                    shard=k,
                    attempt=a,
                    init_idx=init_idx,
                    shift_cm=round(100 * float(np.linalg.norm(shift)), 2),
                    inflation_m=W.INFLATE_M,
                    payload_r_m=R.PAYLOAD_SPHERE_R,
                    handoff_up_cm=rec.get("handoff_up_cm"),
                    handoff_out_cm=rec.get("handoff_out_cm"),
                    tilt_deg=d0.get("tilt_deg"),
                    n_via=int(len(c)),
                    planned_via_min_cm=round(100 * float(c.min()), 3) if len(c) else None,
                    executor_ok=bool(ok),
                    executor_reason=why,
                    steps=int(t),
                    executed_min_gjk_cm=round(100 * clr, 3),
                    arrival=arrival_errors(wp, ep),
                    frames=int(len(ep.base)),
                )
                with open(os.path.join(CLIPS, nm + ".json"), "w") as fh:
                    json.dump(meta, fh, indent=1, default=str)
                print(
                    f"  {nm}.mp4  {len(ep.base)} frames  planned-via min "
                    f"{meta['planned_via_min_cm']} cm  executed min {meta['executed_min_gjk_cm']}"
                    f" cm  {why}  ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
            finally:
                if env is not None:
                    with contextlib.suppress(Exception):
                        env.close()
    print("RENDER_STEP4B_DONE")


if __name__ == "__main__":
    main()
