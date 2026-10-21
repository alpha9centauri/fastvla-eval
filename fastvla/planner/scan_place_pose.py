"""True payload-vs-obstacle GJK clearance at dgcore's place pose, per scene.

This is the ceiling on certification: the carried bowl's clearance to the declared obstacles at the
pose where it is released. If it is 0, no margin above 0 can certify that scene no matter how good
the planner is. Scene build plus GJK only -- no planning, no execution -- so all 129 are affordable.

Author: Varun Chandra.  Env: planner, GPU 1 only.  N=129
"""
import os, sys, numpy as np
sys.path.insert(0,'/data_new/vchandr2/fastvla-eval')
for p in ('/data_new/vchandr2/vla-safety-interp/src/displacement',
          '/data_new/vchandr2/vla-safety-interp/src/displacement/datagen','/data_new/vchandr2',
          '/data_new/vchandr2/vla-safety-interp/src/physics',
          '/data_new/vchandr2/fastvla-eval/fastvla/labeler'):
    sys.path.insert(0,p)
import dgcore as D, geom_shift, fastgjk, json
from fastvla.planner import world as W
base = {k: np.asarray(v) for k,v in dict(np.load('/data_new/vchandr2/vla-safety-interp/src/displacement/out_baseline_oa1_50/baseline_17500.npz', allow_pickle=True)).items() if k.endswith('__eef')}
obstacle=D.SPEC["names"]["oa1"]["obstacle"]; lo,hi=D.CONF["INIT_IDX_MIN"],D.CONF["INIT_IDX_MAX"]
rows=json.load(open('/data_new/vchandr2/fastvla-eval/fastvla/planner/out/step4b_full.json'))
_n=int(os.environ.get("N","20"))
tg=sorted({(r['shard'],r['attempt']) for r in rows})
tg=tg if _n<=0 else tg[:_n]
import collections
by=collections.defaultdict(list)
for k,a in tg: by[k].append(a)
res=[]
for k in sorted(by):
    want=set(by[k]); rng=np.random.RandomState(131*k)
    for a in range(max(want)+1):
        ii=int(rng.randint(lo,hi+1))
        env=None
        try:
            env,dom,mon,sg,task,init,obs = D.make_env("oa1", ii, res=84); D.set_render(dom,False)
            for _ in range(D.CONF["SETTLE_STEPS"]): obs,*_=env.step([0.0]*6+[-1.0])
            cand=None
            for _ in range(max(1,2*D.CONF["SCENE_TRIES_PER_SIGN"])):
                i_=float(rng.uniform(D.CONF["EVAL_RULE_INTR_MIN_M"],D.CONF["EVAL_RULE_INTR_MAX_M"]))
                c=D.eval_rule_shift(sg,obstacle,base[f"oa1__i{ii:02d}__eef"],i_)
                if c.get("feasible"): cand=c; break
            if a not in want or cand is None: continue
            shift=np.asarray(cand["shift"],float)
            _,_,anch=geom_shift.obstacle_anchors(dom); st=geom_shift.apply_shift(dom,anch,init[ii],shift)
            dom,mon,sg,obs,_=D.setup_scene(env,st,ii,shift=shift,render=False)
            home=np.asarray(obs["robot0_eef_pos"],float); hq=np.asarray(obs["robot0_eef_quat"],float)
            wp=D.nominal_plan(sg,dom,"oa1",home,hq)
            place=next(w for w in wp if str(w[2]).startswith("place:"))
            grasp=next(w for w in wp if str(w[2]).startswith("grasp:"))
            gp=np.asarray(grasp[0],float); pp=np.asarray(place[0],float)
            m=dom.sim.model
            cg=[g for g in range(m.ngeom) if (m.geom_id2name(g) or '').startswith("akita_black_bowl_1")]
            pay=[(mon.dist_engine._w(dom.sim.data,g)[0]-gp[None,:], mon.dist_engine._w(dom.sim.data,g)[1]) for g in cg]
            best={}
            for nm,gs in (("bottle",list(mon.obstacle_geoms[obstacle])),
                          ("box",list(mon.obstacle_geoms.get("white_place_box_1",[])))):
                b=np.inf
                for v_rel,pad in pay:
                    vv=np.ascontiguousarray(v_rel+pp[None,:])
                    for og in gs:
                        Bv,Bpad=mon.dist_engine._w(dom.sim.data,og)
                        d=max(0.0,fastgjk.gjk_distance(vv,Bv)-pad-Bpad)
                        if d<b: b=d
                best[nm]=b
            both=min(best.values())
            res.append(dict(shard=k,attempt=a,init=ii,shift_cm=round(100*float(np.linalg.norm(shift)),2),
                            bottle_cm=round(100*best["bottle"],3), box_cm=round(100*best["box"],3),
                            min_cm=round(100*both,3)))
            print(f"  s{k:02d}a{a:02d} i{ii:02d} shift {res[-1]['shift_cm']:>6}cm  "
                  f"bowl->bottle {res[-1]['bottle_cm']:>6} cm  bowl->box {res[-1]['box_cm']:>6} cm  "
                  f"min {res[-1]['min_cm']:>6} cm", flush=True)
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass
v=np.array([r['min_cm'] for r in res])
print()
print(f"TRUE payload clearance AT THE PLACE POSE over {len(v)} scenes:")
print(f"  min {v.min():.2f}  median {np.median(v):.2f}  max {v.max():.2f} cm")
for thr in (0.5,1.0,1.2,1.5,2.0):
    print(f"  scenes with >= {thr:.1f} cm: {int((v>=thr).sum())}/{len(v)}")
out=os.environ.get("OUTJSON","/data_new/vchandr2/fastvla-eval/fastvla/planner/out/place_pose_gap.json")
json.dump(res, open(out,'w'), indent=1)
clear=[r for r in res if r['min_cm']>=1.0]
print(f"scenes with place-pose clearance >= 1 cm: {len(clear)}/{len(res)}")
print("TARGETS=" + ",".join(f"{r['shard']}:{r['attempt']}" for r in clear))
