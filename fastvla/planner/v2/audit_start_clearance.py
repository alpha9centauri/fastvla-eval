"""Does the candidate placement overlap the robot at its START pose? CPU, no policy, no GPU.

Rule v2 required the stack to clear the bowl's start, the plate and the place pose, and to intrude
at least 3 cm into the recorded EE or link5 path. It never required the stack to clear the robot's
OWN initial configuration -- and the recorded path BEGINS at that configuration, so ranking by
depth of intrusion actively selects positions that sit inside the arm at rest.

This measures, for all 80 candidates, the true GJK distance between the watched robot geoms and
the moved stack at the initial state, before any action is taken. Negative = the scene starts in
penetration and anything the rollout then scores is an artefact of the simulator resolving it.

Author: Varun Chandra.
"""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data_new/vchandr2/rollout_stubs")
DISP = "/data_new/vchandr2/vla_old_research/vla-safety-interp/src/displacement"
for p in (DISP, os.path.join(DISP, "datagen"),
          "/data_new/vchandr2/vla_old_research/vla-safety-interp/src/physics"):
    sys.path.insert(0, p)
import dgcore as D, geom_shift, mujoco                      # noqa: E402
from physics_safety import SafetyMonitor                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cands = json.load(open(os.path.join(HERE, "candidates_v2.json")))["rows"]
rows = {r["cid"]: r for r in json.load(
    open("/data_new/vchandr2/data/planner_v2/step2/rows_step2.json")) if not r.get("skipped")}
out = []
by_init = {}
for c in cands:
    by_init.setdefault(c["init"], []).append(c)
for ii in sorted(by_init):
    ordered = sorted(by_init[ii], key=lambda r: -r["intrude_cm"])
    env, dom, mon, sg, task, init, obs = D.make_env("oa1", ii, res=84)
    for _ in range(D.CONF["SETTLE_STEPS"]):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    _, _, anch = geom_shift.obstacle_anchors(dom)
    m2 = SafetyMonitor(dom, clearance_every=1)
    for j, c in enumerate(ordered):
        st = geom_shift.apply_shift(dom, anch, init[ii], np.asarray(c["shift"], float))
        env.set_init_state(st)
        mujoco.mj_forward(dom.sim.model._model, dom.sim.data._data)
        best, pair = 1e9, None
        for g in m2.idx.robot_geoms:
            for o, gs in m2.obstacle_geoms.items():
                for og in gs:
                    d = mujoco.mj_geomDistance(dom.sim.model._model, dom.sim.data._data,
                                               g, og, 1.0, None)
                    if d < best:
                        best, pair = d, (m2.idx.geom_label(g), m2.idx.geom_label(og))
        cid = f"oa1__i{ii:02d}__c{j}"
        r = rows.get(cid, {})
        out.append(dict(cid=cid, init=ii, bin=c["bin"], intrude_cm=c["intrude_cm"],
                        start_clearance_cm=round(100 * float(best), 3),
                        start_pair=list(pair) if pair else None,
                        scored_collision=r.get("collision"),
                        collision_step=r.get("collision_step"),
                        completion=r.get("completion")))
        print(f"  {cid}  {c['bin']:<8} intr {c['intrude_cm']:5.2f}  start "
              f"{100*best:8.3f} cm  scored {r.get('collision')}  step {r.get('collision_step')}",
              flush=True)
    env.close()
json.dump(out, open(os.path.join(HERE, "start_clearance.json"), "w"), indent=1)
neg = [r for r in out if r["start_clearance_cm"] < 0]
print(f"\n{len(neg)} of {len(out)} candidates START IN PENETRATION")
print(f"  of those, scored a collision: {sum(bool(r['scored_collision']) for r in neg)}")
clean = [r for r in out if r["start_clearance_cm"] >= 0]
print(f"{len(clean)} start clear; of those, scored a collision: "
      f"{sum(bool(r['scored_collision']) for r in clean)}")
for r in clean:
    if r["scored_collision"]:
        print(f"    GENUINE: {r['cid']}  {r['bin']}  intrusion {r['intrude_cm']:.2f} cm  "
              f"start clearance {r['start_clearance_cm']:.3f} cm  step {r['collision_step']}")
print("AUDIT_DONE")
