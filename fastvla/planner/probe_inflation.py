"""How many scenes still plan, and are the planned vias >= 2 cm clear? -- step 4b probe.

Runs the N cheapest-to-reproduce scenes that returned a path in step 4, under the current
inflation and payload settings, and measures the carried payload's GJK clearance at every planned
via against the UNINFLATED geometry through the labeler's own engine. That is the check item 1
asks for: a cuRobo-clear plan should be >= 2 cm clear to the certifier.

Author: Varun Chandra
Env: planner. GPU 1 only. N=8 PLANNER_INFLATE_M=... PLANNER_PAYLOAD_SPHERE_R=...
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
obstacle = D.SPEC["names"]["oa1"]["obstacle"]; lo,hi = D.CONF["INIT_IDX_MIN"], D.CONF["INIT_IDX_MAX"]
rows=json.load(open('/data_new/vchandr2/fastvla-eval/fastvla/planner/out/step4_rows.json'))
# the first N scenes that returned a path in step 4, cheapest first (low attempt index)
cand=[r for r in rows if r['plan_ok']]
cand.sort(key=lambda r:(r['attempt'], r['shard']))
N=int(os.environ.get("N","8"))
picks=cand[:N]
print(f"probing {len(picks)} of the {len(cand)} step-4 planned scenes, inflate="
      f"{W.INFLATE_M} payload r={R.PAYLOAD_SPHERE_R} budget={R.PAYLOAD_SPHERE_BUDGET}")
nplan=0; mins=[]
for r in picks:
    k,a = r['shard'], r['attempt']
    rng=np.random.RandomState(0+131*k)
    for i in range(a):
        ii=int(rng.randint(lo,hi+1)); D.generate_demo("oa1",ii,"varied",rng,res=84,placement="eval_rule",baseline=base)
    init_idx=int(rng.randint(lo,hi+1))
    env, dom, mon, sg, task, init, obs = D.make_env("oa1", init_idx, res=84); D.set_render(dom,False)
    for _ in range(D.CONF["SETTLE_STEPS"]): obs,*_=env.step([0.0]*6+[-1.0])
    cd=None
    for _ in range(max(1,2*D.CONF["SCENE_TRIES_PER_SIGN"])):
        i_=float(rng.uniform(D.CONF["EVAL_RULE_INTR_MIN_M"],D.CONF["EVAL_RULE_INTR_MAX_M"]))
        c=D.eval_rule_shift(sg,obstacle,base[f"oa1__i{init_idx:02d}__eef"],i_)
        if c.get("feasible"): cd=c; break
    if cd is None: env.close(); continue
    shift=np.asarray(cd["shift"],float)
    _,_,anch=geom_shift.obstacle_anchors(dom); st=geom_shift.apply_shift(dom,anch,init[init_idx],shift)
    dom,mon,sg,obs,_=D.setup_scene(env,st,init_idx,shift=shift,render=False)
    home=np.asarray(obs["robot0_eef_pos"],float); hq=np.asarray(obs["robot0_eef_quat"],float)
    wp0=D.nominal_plan(sg,dom,"oa1",home,hq)
    wp,det,shape=D.plan_detours(sg,obstacle,wp0,ceiling=D.measure_ceiling("oa1")["usable_m"])
    vias=[w for w in wp if str(w[2]).startswith("detour_curobo")]
    if not shape.get("plan_clears") or not vias:
        print(f"  s{k:02d}a{a:02d} i{init_idx:02d} shift {100*np.linalg.norm(shift):5.2f}cm  NO PLAN ({shape.get('reason')})")
        env.close(); continue
    nplan+=1
    hull=mon._obstacle_hull_world(dom.sim.data,obstacle); ogeoms=list(mon.obstacle_geoms[obstacle])
    cg=[g for g in range(dom.sim.model.ngeom) if (dom.sim.model.geom_id2name(g) or '').startswith("akita_black_bowl_1")]
    grasp=next(w for w in wp if str(w[2]).startswith("grasp:")); gp=np.asarray(grasp[0],float)
    pay=[(mon.dist_engine._w(dom.sim.data,g)[0]-gp[None,:], mon.dist_engine._w(dom.sim.data,g)[1]) for g in cg]
    ds=[]
    for w in vias:
        p=np.asarray(w[0],float); best=np.inf
        for v_rel,pad in pay:
            vv=np.ascontiguousarray(v_rel+p[None,:])
            for og in ogeoms:
                Bv,Bpad=mon.dist_engine._w(dom.sim.data,og)
                d=max(0.0,fastgjk.gjk_distance(vv,Bv)-pad-Bpad)
                if d<best: best=d
        ds.append(best)
    ds=np.asarray(ds); mins.append(float(ds.min()))
    print(f"  s{k:02d}a{a:02d} i{init_idx:02d} shift {100*np.linalg.norm(shift):5.2f}cm  "
          f"vias {len(vias):>3}  planned-via min GJK {100*ds.min():6.2f} cm  "
          f"{'>=2cm OK' if ds.min()>=0.02 else 'UNDER 2cm'}")
    env.close()
print()
print(f"planned {nplan}/{len(picks)}")
if mins:
    m=np.asarray(mins)
    print(f"planned-via min clearance: min {100*m.min():.2f} median {100*np.median(m):.2f} max {100*m.max():.2f} cm; >=2cm on {int((m>=0.02).sum())}/{len(m)}")
