"""Step 4b: the inflation sweep, and the full run at the chosen setting.

Two modes over the same scene set and the same seeds as step 4:

  MODE=plan   build the scene, run the route stage only, and measure the carried payload's GJK
              clearance at every planned via against the UNINFLATED geometry. No execution, so a
              sweep over inflation settings is cheap.
  MODE=full   call dgcore.generate_demo, so the real certifier runs: dry pass, 2 cm GJK gate and
              the rendered-replay gate, exactly as always.

Scene identity. Step 4 drew scenes as `gen_stagec.py` does: shard k seeded SEED + 131*k, one
randint for the init index, then uniform draws until eval_rule_shift reports feasible. That
consumption depends only on the scene geometry -- the planner runs afterwards and never touches
the rng -- so a shard can be walked forward cheaply by building each scene and running the draw
loop, without executing anything. Walking each shard ONCE and handling every target attempt along
the way is what makes 129 scenes affordable; replaying per scene would re-run every predecessor.

Author: Varun Chandra
Env: planner. GPU 1 only.
Env vars: MODE=plan|full  TARGETS=auto|<n>  N=24  PLANNER_INFLATE_M=...  RES=256
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
           "/data_new/vchandr2/fastvla-eval",
           "/data_new/vchandr2/vla-safety-interp/src/physics",
           "/data_new/vchandr2/fastvla-eval/fastvla/labeler"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D                                  # noqa: E402
import fastgjk                                      # noqa: E402
import geom_shift                                   # noqa: E402
from fastvla.planner import route as R               # noqa: E402
from fastvla.planner import world as W               # noqa: E402

TAG = "oa1"
SEED = int(os.environ.get("SEED", "0"))
MODE = os.environ.get("MODE", "plan")
N = int(os.environ.get("N", "24"))
RES = int(os.environ.get("RES", "84" if os.environ.get("MODE", "plan") == "plan" else "256"))
FLOOR_M = float(os.environ.get("FLOOR_M", "0.01"))
TAGOUT = os.environ.get("TAGOUT", "")
BASELINE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")
CARRIED = "akita_black_bowl_1"


def targets():
    """The step-4 scenes that returned a path, in run order, first N of them."""
    rows = json.load(open(os.path.join(OUT, "step4_rows.json")))
    ok = [(r["shard"], r["attempt"]) for r in rows if r["plan_ok"]]
    ok.sort()
    return ok[:N] if N > 0 else ok


def draw_scene(sg, obstacle, traj, rng):
    """generate_demo's eval-rule draw loop. Advances the rng exactly as the real run does."""
    for _ in range(max(1, 2 * D.CONF["SCENE_TRIES_PER_SIGN"])):
        i_ = float(rng.uniform(D.CONF["EVAL_RULE_INTR_MIN_M"], D.CONF["EVAL_RULE_INTR_MAX_M"]))
        cand = D.eval_rule_shift(sg, obstacle, traj, i_)
        if cand.get("feasible"):
            return cand, i_
    return None, None


def via_clearances(dom, mon, wp, obstacle):
    """Payload-vs-obstacle GJK clearance at every planned via, against UNINFLATED geometry.

    The payload hull is carried rigidly from the grasp pose, and the comparison runs through the
    labeler's own engine against the obstacle's convex pieces -- the certifier's geometry, not the
    planner's.
    """
    vias = [w for w in wp if str(w[2]).startswith("detour_curobo")]
    if not vias:
        return np.asarray([])
    m = dom.sim.model
    cg = [g for g in range(m.ngeom) if (m.geom_id2name(g) or "").startswith(CARRIED)]
    grasp = next(w for w in wp if str(w[2]).startswith("grasp:"))
    gp = np.asarray(grasp[0], float)
    pay = []
    for g in cg:
        v, pad = mon.dist_engine._w(dom.sim.data, g)
        pay.append((v - gp[None, :], pad))
    ogeoms = list(mon.obstacle_geoms[obstacle])
    out = []
    for w in vias:
        p = np.asarray(w[0], float)
        best = np.inf
        for v_rel, pad in pay:
            vv = np.ascontiguousarray(v_rel + p[None, :])
            for og in ogeoms:
                Bv, Bpad = mon.dist_engine._w(dom.sim.data, og)
                d = max(0.0, fastgjk.gjk_distance(vv, Bv) - pad - Bpad)
                if d < best:
                    best = d
        out.append(best)
    return np.asarray(out)


def arrival_errors(wp, ep):
    """Per-waypoint arrival error for an executed episode: how close the EE got to each via."""
    if not len(getattr(ep, "eef", [])):
        return []
    E = np.asarray(ep.eef)
    rows = []
    for w in wp:
        p = np.asarray(w[0], float)
        d = np.linalg.norm(E - p[None, :], axis=1)
        rows.append(dict(label=str(w[2]), closest_mm=round(1000 * float(d.min()), 2)))
    return rows


def main():
    R.install("curobo")
    tg = targets()
    by_shard = collections.defaultdict(list)
    for k, a in tg:
        by_shard[k].append(a)
    print(f"MODE={MODE}  targets={len(tg)}  inflation={W.INFLATE_M} support={W.INFLATE_SUPPORT_M} "
          f"payload r={R.PAYLOAD_SPHERE_R} budget={R.PAYLOAD_SPHERE_BUDGET}  res={RES}",
          flush=True)
    base = {k: np.asarray(v) for k, v in
            dict(np.load(BASELINE, allow_pickle=True)).items() if k.endswith("__eef")}
    obstacle = D.SPEC["names"][TAG]["obstacle"]
    lo, hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
    if MODE == "full":
        D.CONF["CERTIFY_MIN_M"] = FLOOR_M

    rows, t0 = [], time.perf_counter()
    for k in sorted(by_shard):
        want = set(by_shard[k])
        rng = np.random.RandomState(SEED + 7919 * 0 + 131 * k)
        for a in range(max(want) + 1):
            init_idx = int(rng.randint(lo, hi + 1))
            env = None
            try:
                env, dom, mon, sg, task, init, obs = D.make_env(TAG, init_idx, res=RES)
                D.set_render(dom, MODE == "full")
                for _ in range(D.CONF["SETTLE_STEPS"]):
                    obs, *_ = env.step([0.0] * 6 + [-1.0])
                cand, intr = draw_scene(sg, obstacle, base[f"{TAG}__i{init_idx:02d}__eef"], rng)
                if a not in want:
                    continue            # rng is now where the real run left it; nothing else
                if cand is None:
                    rows.append(dict(shard=k, attempt=a, init_idx=init_idx, planned=False,
                                     reason="scene_reject:no_feasible_placement"))
                    continue
                shift = np.asarray(cand["shift"], float)
                if MODE == "full":
                    env.close()
                    env = None
                    rng2 = np.random.RandomState(0)      # unused by eval_rule after the draw
                    rr = run_full(init_idx, shift, base, obstacle, rng2)
                    rr.update(shard=k, attempt=a)
                    rows.append(rr)
                else:
                    rr = run_plan(env, dom, mon, sg, init, init_idx, shift, obstacle)
                    rr.update(shard=k, attempt=a, init_idx=init_idx,
                              shift_cm=round(100 * float(np.linalg.norm(shift)), 3))
                    rows.append(rr)
                r = rows[-1]
                print(f"  s{k:02d}a{a:02d} i{init_idx:02d} shift {r.get('shift_cm')}cm "
                      f"planned={r.get('planned')} "
                      f"via_min={r.get('via_min_cm')} {r.get('reason') or ''} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
                json.dump(rows, open(os.path.join(
                    OUT, f"step4b_{MODE}{TAGOUT}.json"), "w"), indent=1, default=str)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
    report(rows, time.perf_counter() - t0)


def run_plan(env, dom, mon, sg, init, init_idx, shift, obstacle):
    _, _, anch = geom_shift.obstacle_anchors(dom)
    st = geom_shift.apply_shift(dom, anch, init[init_idx], shift)
    dom, mon, sg, obs, _ = D.setup_scene(env, st, init_idx, shift=shift, render=False)
    home = np.asarray(obs["robot0_eef_pos"], float)
    hq = np.asarray(obs["robot0_eef_quat"], float)
    wp0 = D.nominal_plan(sg, dom, TAG, home, hq)
    R.reset_stats()
    wp, det, shape = D.plan_detours(sg, obstacle, wp0,
                                    ceiling=D.measure_ceiling(TAG)["usable_m"])
    plan = next((s for s in R.stats() if s.get("stage") == "plan"), {}) or {}
    if not shape.get("plan_clears"):
        return dict(planned=False, reason=shape.get("reason"),
                    plan_seconds_total=plan.get("seconds"))
    c = via_clearances(dom, mon, wp, obstacle)
    d0 = (det or [{}])[0] if det else {}
    h = d0.get("handoff") or {}
    return dict(planned=True, reason=None, n_via=int(len(c)),
                via_min_cm=round(100 * float(c.min()), 3) if len(c) else None,
                via_median_cm=round(100 * float(np.median(c)), 3) if len(c) else None,
                via_all_cm=[round(100 * float(x), 3) for x in c],
                plan_seconds_total=plan.get("seconds"),
                plan_seconds_single=plan.get("seconds_single"),
                handoff_up_cm=(round(100 * h["up_m"], 1) if h.get("up_m") is not None else None),
                handoff_out_cm=(round(100 * h["out_m"], 1) if h.get("out_m") is not None else None),
                tilt_deg=d0.get("tilt_deg"))


def run_full(init_idx, shift, base, obstacle, rng):
    """generate_demo on a scene whose shift is already decided, through the real certifier."""
    env = None
    try:
        env, dom, mon, sg, task, init, obs = D.make_env(TAG, init_idx, res=RES)
        D.set_render(dom, True)
        for _ in range(D.CONF["SETTLE_STEPS"]):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        _, _, anch = geom_shift.obstacle_anchors(dom)
        st = geom_shift.apply_shift(dom, anch, init[init_idx], shift)
        dom, mon, sg, obs, _ = D.setup_scene(env, st, init_idx, shift=shift, render=True)
        home = np.asarray(obs["robot0_eef_pos"], float)
        hq = np.asarray(obs["robot0_eef_quat"], float)
        wp0 = D.nominal_plan(sg, dom, TAG, home, hq)
        R.reset_stats()
        wp, det, shape = D.plan_detours(sg, obstacle, wp0,
                                        ceiling=D.measure_ceiling(TAG)["usable_m"])
        plan = next((s for s in R.stats() if s.get("stage") == "plan"), {}) or {}
        out = dict(init_idx=init_idx, shift_cm=round(100 * float(np.linalg.norm(shift)), 3),
                   plan_seconds_total=plan.get("seconds"),
                   plan_seconds_single=plan.get("seconds_single"))
        if not shape.get("plan_clears"):
            out.update(planned=False, reason=shape.get("reason"), accepted=False)
            return out
        c = via_clearances(dom, mon, wp, obstacle)
        d0 = (det or [{}])[0] if det else {}
        h = d0.get("handoff") or {}
        out.update(planned=True, n_via=int(len(c)),
                   via_min_cm=round(100 * float(c.min()), 3) if len(c) else None,
                   via_median_cm=round(100 * float(np.median(c)), 3) if len(c) else None,
                   via_all_cm=[round(100 * float(x), 3) for x in c],
                   handoff_up_cm=(round(100 * h["up_m"], 1) if h.get("up_m") is not None else None),
                   handoff_out_cm=(round(100 * h["out_m"], 1) if h.get("out_m") is not None else None),
                   tilt_deg=d0.get("tilt_deg"))
        # the real thing: execute, certify, replay-gate
        ep = D.Episode()
        ep.render = True
        ep.last_obs = obs
        mon.attach()
        t, ok, why = D.execute(env, dom, mon, wp, ep, t0=0, ref_quat=hq)
        mon.detach()
        clr = float(np.min(ep.clearance)) if len(ep.clearance) else float("nan")
        out.update(executor_ok=bool(ok), executor_reason=why, steps=int(t),
                   dry_min_clearance_cm=round(100 * clr, 3),
                   success=bool(env.check_success()))
        if not ok:
            out.update(accepted=False, reason=f"executor:{why}",
                       arrival=arrival_errors(wp, ep))
            return out
        if not out["success"]:
            out.update(accepted=False, reason="success_predicate_false")
            return out
        out.update(accepted=bool(100 * clr >= 100 * FLOOR_M),
                   reason=None if 100 * clr >= 100 * FLOOR_M else "certification")
        np.savez_compressed(os.path.join(OUT, f"step4b_ep_i{init_idx:02d}.npz"),
                            base=np.asarray(ep.base, np.uint8), clearance=np.asarray(ep.clearance),
                            eef=np.asarray(ep.eef), meta=json.dumps(out, default=str))
        return out
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def report(rows, wall):
    pl = [r for r in rows if r.get("planned")]
    allv = np.asarray([v for r in pl for v in (r.get("via_all_cm") or [])], float)
    mins = np.asarray([r["via_min_cm"] for r in pl if r.get("via_min_cm") is not None], float)
    L = []
    A = L.append
    A("=" * 92)
    A(f"STEP 4B {MODE.upper()} -- inflation {W.INFLATE_M} m, payload r={R.PAYLOAD_SPHERE_R} "
      f"budget={R.PAYLOAD_SPHERE_BUDGET}")
    A("=" * 92)
    A(f"  scenes                {len(rows)}      wall {wall / 60:.1f} min")
    A(f"  returned a path       {len(pl)}/{len(rows)} = {100.0 * len(pl) / max(1, len(rows)):.1f}%")
    if len(allv):
        A(f"  planned-via clearance, ALL vias ({len(allv)}): min {allv.min():.2f} "
          f"median {np.median(allv):.2f} max {allv.max():.2f} cm")
    if len(mins):
        A(f"  per-scene min via clearance: min {mins.min():.2f} median {np.median(mins):.2f} "
          f"max {mins.max():.2f} cm")
        A(f"  planned scenes with min via >= 2 cm: {(mins >= 2.0).sum()}/{len(mins)} = "
          f"{100.0 * (mins >= 2.0).mean():.1f}%")
        A(f"  planned scenes with min via >= 1 cm: {(mins >= 1.0).sum()}/{len(mins)} = "
          f"{100.0 * (mins >= 1.0).mean():.1f}%")
    if MODE == "full":
        a1 = [r for r in rows if r.get("accepted")]
        a2 = [r for r in a1 if (r.get("dry_min_clearance_cm") or 0) >= 2.0]
        A(f"  ACCEPTED at 1 cm      {len(a1)}/{len(rows)}")
        A(f"  ACCEPTED at 2 cm      {len(a2)}/{len(rows)}")
        c = collections.Counter(
            ("accepted" if r.get("accepted") else (r.get("reason") or "?").split(":")[0])
            for r in rows)
        A("  outcome breakdown:")
        for kk, n in c.most_common():
            A(f"    {kk:<40} {n:>4}")
        bad = [r for r in rows if r.get("arrival")]
        if bad:
            A("  executor failures -- worst per-waypoint arrival error:")
            for r in bad[:12]:
                w = max(r["arrival"], key=lambda x: x["closest_mm"])
                A(f"    i{r['init_idx']:02d} {r.get('executor_reason', ''):<34} "
                  f"worst {w['label'][:28]:<28} {w['closest_mm']:>8.1f} mm")
    txt = "\n".join(L)
    print()
    print(txt)
    open(os.path.join(OUT, f"step4b_{MODE}{TAGOUT}_report.txt"), "w").write(txt + "\n")
    print(f"STEP4B_{MODE.upper()}_DONE")


if __name__ == "__main__":
    main()
