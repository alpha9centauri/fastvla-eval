"""Planned vias vs executed path: where is the clearance lost?

Step 4 measured all 129 scenes with a path landing at a dry-pass GJK clearance of 0.000 or
0.072 cm -- contact every time. Two values across 129 scenes is not a tracking distribution, so
this evaluates the clearance at the PLANNED vias instead: the carried payload's hull, carried
rigidly from the grasp pose, against the obstacle's 244 convex pieces through the labeler's own
GJK engine. If the planned vias are clear by a margin the executor is at fault; if they are not,
the plan never satisfied the certification floor and the executor is a separate problem.

Result on s09a00 (init 25, 11.47 cm shift, plan reported clear): min 0.39 cm over 24 vias, 3 under
2 cm. The plan itself is not 2 cm clear. n=1 scene -- see RESULTS.md.

Author: Varun Chandra
Env: planner. GPU 1 only.
"""
import os, sys, json, numpy as np
sys.path.insert(0,'/data_new/vchandr2/fastvla-eval')
for p in ('/data_new/vchandr2/vla-safety-interp/src/displacement',
          '/data_new/vchandr2/vla-safety-interp/src/displacement/datagen','/data_new/vchandr2',
          '/data_new/vchandr2/vla-safety-interp/src/physics',
          '/data_new/vchandr2/fastvla-eval/fastvla/labeler'):
    sys.path.insert(0,p)
import dgcore as D, geom_shift, fastgjk
from fastvla.planner import world as W, route as R
R.install("curobo")
base = {k: np.asarray(v) for k,v in dict(np.load('/data_new/vchandr2/vla-safety-interp/src/displacement/out_baseline_oa1_50/baseline_17500.npz', allow_pickle=True)).items() if k.endswith('__eef')}
obstacle = D.SPEC["names"]["oa1"]["obstacle"]
lo,hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
import os
SCENES=[tuple(int(x) for x in t.split(":")) for t in os.environ.get("SCENES","9:0").split(",")]
for (k,a) in SCENES:
    rng = np.random.RandomState(0 + 131*k)
    for i in range(a):
        ii=int(rng.randint(lo,hi+1)); D.generate_demo("oa1",ii,"varied",rng,res=84,placement="eval_rule",baseline=base)
    init_idx=int(rng.randint(lo,hi+1))
    env, dom, mon, sg, task, init, obs = D.make_env("oa1", init_idx, res=84); D.set_render(dom, False)
    for _ in range(D.CONF["SETTLE_STEPS"]): obs,*_ = env.step([0.0]*6+[-1.0])
    cand=None
    for _ in range(max(1,2*D.CONF["SCENE_TRIES_PER_SIGN"])):
        i_=float(rng.uniform(D.CONF["EVAL_RULE_INTR_MIN_M"],D.CONF["EVAL_RULE_INTR_MAX_M"]))
        c=D.eval_rule_shift(sg,obstacle,base[f"oa1__i{init_idx:02d}__eef"],i_)
        if c.get("feasible"): cand=c; break
    shift=np.asarray(cand["shift"],float)
    _,_,anch=geom_shift.obstacle_anchors(dom); st=geom_shift.apply_shift(dom,anch,init[init_idx],shift)
    dom,mon,sg,obs,_ = D.setup_scene(env, st, init_idx, shift=shift, render=False)
    home=np.asarray(obs["robot0_eef_pos"],float); hq=np.asarray(obs["robot0_eef_quat"],float)
    wp0=D.nominal_plan(sg,dom,"oa1",home,hq)
    wp,det,shape = D.plan_detours(sg,obstacle,wp0,ceiling=D.measure_ceiling("oa1")["usable_m"])
    print(f"init {init_idx} shift {100*np.linalg.norm(shift):.2f} cm  plan_clears={shape.get('plan_clears')}")
    vias=[w for w in wp if str(w[2]).startswith("detour_curobo")]
    print(f"planned detour vias: {len(vias)}")
    # GJK clearance of the PAYLOAD HULL at each planned via, carried rigidly from the grasp
    hull = mon._obstacle_hull_world(dom.sim.data, obstacle)
    ogeoms = list(mon.obstacle_geoms[obstacle])
    carried = "akita_black_bowl_1"
    cg = [g for g in range(dom.sim.model.ngeom)
          if (dom.sim.model.geom_id2name(g) or '').startswith(carried)]
    grasp = next(w for w in wp if str(w[2]).startswith("grasp:"))
    gp = np.asarray(grasp[0], float)
    pay_rel = []
    for g in cg:
        v,pad = mon.dist_engine._w(dom.sim.data, g)
        pay_rel.append((v - gp[None,:], pad))
    ds=[]
    for w in vias:
        p = np.asarray(w[0], float)
        best = np.inf
        for v_rel, pad in pay_rel:
            vv = v_rel + p[None,:]
            for og in ogeoms:
                Bv,Bpad = mon.dist_engine._w(dom.sim.data, og)
                d = max(0.0, fastgjk.gjk_distance(np.ascontiguousarray(vv), Bv) - pad - Bpad)
                if d < best: best = d
        ds.append(best)
    ds=np.asarray(ds)
    print(f"  payload-vs-obstacle GJK clearance AT THE PLANNED VIAS: min {100*ds.min():.2f} cm, "
          f"median {100*np.median(ds):.2f} cm, max {100*ds.max():.2f} cm")
    print(f"  vias with clearance < 2 cm: {(ds<0.02).sum()}/{len(ds)};  < 1 cm: {(ds<0.01).sum()}")
    env.close()
