"""Render a certification-refused scene: two views, plus the frame where contact happens.

One of the 11 feasible scenes that planned, executed to the end and satisfied the task's success
predicate, and was still refused because the minimum GJK clearance reached 0.00 cm. Those are the
interesting failures: the demo "works" and is still not certifiable.

Three outputs:
  <name>_agentview.mp4   the standard view, H.264
  <name>_birdview.mp4    from above. `birdview` is a MuJoCo camera but not an env observable, so
                         it is rendered straight from the sim per control step rather than added
                         to the observation dict, which would change what the executor sees.
  mincontact_...png      the frame at minimum clearance, both views side by side, with the
                         colliding pair named in the filename.

The contact pair comes from the monitor itself: `SafetyMonitor.end_step` is wrapped so every
record is captured as dgcore's own executor produces it. Nothing in the execution path is
reimplemented.

Frames are checked for the uninitialised-buffer failure before anything is written -- 4 of the
first 8 clips in this milestone were silently noise.

Author: Varun Chandra.  Env: planner, GPU 1 only.  SCENE=4:8  RES=256
"""

import contextlib
import json
import os
import re
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

SCENE = os.environ.get("SCENE", "4:8")
RES = int(os.environ.get("RES", "256"))
NOISE_THRESHOLD = float(os.environ.get("NOISE_THRESHOLD", "40"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))
BASE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")


def frame_score(fr):
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


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "none"))


def run(k, a):
    base = {
        kk: np.asarray(v)
        for kk, v in dict(np.load(BASE, allow_pickle=True)).items()
        if kk.endswith("__eef")
    }
    obstacle = D.SPEC["names"]["oa1"]["obstacle"]
    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    rng = np.random.RandomState(131 * k)
    for i in range(a + 1):
        init_idx = int(rng.randint(lo, hi + 1))
        env = None
        try:
            env, dom, mon, sg, task, init, obs = D.make_env("oa1", init_idx, res=RES)
            D.set_render(dom, True)
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
                return None, dict(reason=f"no plan: {shape.get('reason')}")

            recs, birds = [], []
            orig_end = mon.end_step

            def end_step_capture(orig_end=orig_end, recs=recs, birds=birds, dom=dom):
                rec = orig_end()
                # The colliding pair is NOT at the top level of the record: SafetyMonitor keeps it
                # per obstacle, in rec["per_obstacle"][name], and names it from the contact pair
                # when there is contact and from the GJK argmin when there is not. rec["pairs"]
                # carries the raw contact list.
                closest = rec.get("closest_obstacle")
                po = (rec.get("per_obstacle") or {}).get(closest, {}) if closest else {}
                pairs = rec.get("pairs") or []
                recs.append(
                    dict(
                        min_clearance=rec.get("min_clearance"),
                        contact=rec.get("contact"),
                        closest_obstacle=closest,
                        actor=po.get("actor"),
                        actor_geom=po.get("actor_geom"),
                        obstacle_geom=po.get("obstacle_geom"),
                        obstacles_hit=rec.get("obstacles_hit"),
                        first_pair=(dict(pairs[0]) if pairs else None),
                    )
                )
                try:
                    img = dom.sim.render(width=RES, height=RES, camera_name="birdview")
                    birds.append(np.asarray(img, np.uint8)[::-1].copy())
                except Exception:
                    birds.append(np.zeros((RES, RES, 3), np.uint8))
                return rec

            mon.end_step = end_step_capture
            ep = D.Episode()
            ep.render = True
            ep.last_obs = obs
            mon.attach()
            t, ok, why = D.execute(env, dom, mon, wp, ep, t0=0, ref_quat=hq)
            mon.detach()
            mon.end_step = orig_end

            clr = np.asarray(ep.clearance, float)  # already centimetres
            c = via_clearances(dom, mon, wp, obstacle)
            meta = dict(
                shard=k,
                attempt=a,
                init_idx=init_idx,
                shift_cm=round(100 * float(np.linalg.norm(shift)), 3),
                min_clearance_cm=round(float(clr.min()), 3),
                steps=int(t),
                executor_ok=bool(ok),
                executor_reason=why,
                success=bool(env.check_success()),
                planned_via_min_cm=round(100 * float(c.min()), 3) if len(c) else None,
                tilt_deg=(det or [{}])[0].get("tilt_deg"),
                n_agentview=len(ep.base),
                n_birdview=len(birds),
                n_recs=len(recs),
            )
            return (np.asarray(ep.base, np.uint8), np.asarray(birds, np.uint8), clr, recs), meta
        finally:
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()
    return None, dict(reason="scene not reached")


def main():
    R.install("curobo")
    k, a = (int(x) for x in SCENE.split(":"))
    for attempt in range(1, MAX_TRIES + 1):
        t0 = time.perf_counter()
        got, meta = run(k, a)
        if got is None:
            print(f"  s{k:02d}a{a:02d}: {meta.get('reason')}", flush=True)
            return
        agent, bird, clr, recs = got
        scores = [frame_score(agent[i]) for i in (0, len(agent) // 2, len(agent) - 1)]
        if max(scores) > NOISE_THRESHOLD:
            print(f"  try {attempt}: agentview is NOISE ({max(scores):.1f}), retrying", flush=True)
            continue

        nm = (
            f"refused_s{k:02d}a{a:02d}_i{meta['init_idx']:02d}_"
            f"shift{meta['shift_cm']:.1f}cm_clr{meta['min_clearance_cm']:.2f}cm"
        )
        write_video(os.path.join(CLIPS, nm + "_agentview.mp4"), agent)
        bscore = max(frame_score(bird[i]) for i in (0, len(bird) // 2, len(bird) - 1))
        if bscore <= NOISE_THRESHOLD and bird.any():
            write_video(os.path.join(CLIPS, nm + "_birdview.mp4"), bird)
            meta["birdview"] = "written"
        else:
            meta["birdview"] = f"skipped, score {bscore:.1f}"

        # the moment of minimum clearance, and who was touching whom
        j = int(np.argmin(clr))
        rec = recs[j] if j < len(recs) else {}
        pair = rec.get("first_pair") or {}
        actor_geom = rec.get("actor_geom") or pair.get("actor_geom")
        obs_geom = rec.get("obstacle_geom") or pair.get("obstacle_geom") or pair.get("obstacle")
        meta.update(
            min_at_step=j,
            min_clearance_at_step_cm=round(float(clr[j]), 3),
            contact=bool(rec.get("contact")),
            actor=rec.get("actor"),
            closest_obstacle=rec.get("closest_obstacle"),
            actor_geom=actor_geom,
            obstacle_geom=obs_geom,
            obstacles_hit=rec.get("obstacles_hit"),
            first_pair=pair,
            agentview_frame_scores=[round(s, 2) for s in scores],
        )
        import imageio.v2 as iio

        panel = agent[j]
        if j < len(bird) and bird[j].any():
            panel = np.concatenate([agent[j], bird[j]], axis=1)
        png = (
            f"mincontact_i{meta['init_idx']:02d}_"
            f"{safe(actor_geom)}__vs__{safe(obs_geom)}_"
            f"step{j}_clr{clr[j]:.2f}cm.png"
        )
        iio.imwrite(os.path.join(CLIPS, png), np.ascontiguousarray(panel))
        with open(os.path.join(CLIPS, nm + ".json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        print(f"  {nm}_agentview.mp4  {len(agent)} frames  (score {max(scores):.1f})", flush=True)
        print(f"  birdview: {meta['birdview']}", flush=True)
        print(
            f"  min clearance {meta['min_clearance_cm']} cm at step {j}: "
            f"{actor_geom}  vs  {obs_geom}   closest obstacle "
            f"{rec.get('closest_obstacle')}  hit {rec.get('obstacles_hit')}  "
            f"(contact={rec.get('contact')})",
            flush=True,
        )
        print(f"  {png}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        return
    print("  all tries produced noise")


if __name__ == "__main__":
    main()
