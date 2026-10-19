"""Step 2 verification: does the planner see the same obstacle the certifier sees?

Milestone 7a step 2. The world model is only useful if a plan that the planner calls clear is
clear to the thing that certifies demos. This compares the two collision checkers on the route the
scripted planner actually failed on -- the straight line of the carried traverse, on eval-rule oa1
scenes -- and requires them to agree on two things:

  1. the straight-line route collides
  2. the link that hits is the same link

The comparison is at IDENTICAL JOINT CONFIGURATIONS, which is the only way the question is
well-posed. cuRobo plans in joint space; the labeler measures whatever pose MuJoCo is in. So each
sampled end-effector pose on the straight line is solved to a joint configuration once, with
cuRobo's IK, and then both checkers are asked about that same configuration: cuRobo through
`RobotWorld.get_collision_distance` over its sphere model, the labeler through `fastgjk` over the
MuJoCo geom hulls after `sim.forward()` at the same joint angles.

The two will not agree to the millimetre and are not asked to. cuRobo's arm is 65 spheres fitted
to the Panda meshes; the labeler's arm is the MuJoCo collision hulls. Three distances are reported
per configuration to keep that difference visible rather than hidden in a pass/fail:

  planner   sphere model vs the obstacle union hull        (what the planner decides on)
  labeler   MuJoCo hulls vs the obstacle union hull        (same obstacle, different robot model)
  exact     MuJoCo hulls vs all 244 convex pieces          (what certification actually uses)

Author: Varun Chandra
Env: planner. GPU 1 only.
Env vars: INITS=0,1,2  N_SAMPLES=24  INTRUSION_M=0.05
"""
from __future__ import annotations

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
           os.path.join("/data_new/vchandr2/fastvla-eval", "fastvla", "labeler")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D                                    # noqa: E402
import fastgjk                                        # noqa: E402  the C++ labeler core
from fastvla.planner import world as W                 # noqa: E402

import torch                                          # noqa: E402
from curobo.geom.types import Mesh, WorldConfig        # noqa: E402
from curobo.types.base import TensorDeviceType         # noqa: E402
from curobo.types.math import Pose                     # noqa: E402
from curobo.types.robot import RobotConfig             # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml   # noqa: E402
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig      # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig          # noqa: E402

TAG = "oa1"
INITS = [int(x) for x in os.environ.get("INITS", "0,1,2").split(",")]
N_SAMPLES = int(os.environ.get("N_SAMPLES", "24"))
INTRUSION_M = float(os.environ.get("INTRUSION_M", "0.05"))
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")

# MuJoCo calls them robot0_linkN, cuRobo calls them panda_linkN. The fingers and hand differ in
# naming too; only the arm links matter for the comparison, because the hand is what holds the
# payload and the payload is handled separately.
LINK_ALIASES = {f"robot0_link{i}": f"panda_link{i}" for i in range(8)}
LINK_ALIASES.update({"gripper0_hand": "panda_hand",
                     "gripper0_finger1": "panda_leftfinger",
                     "gripper0_finger2": "panda_rightfinger"})


def canon_link(name):
    """Map either naming scheme onto a common label, so 'which link' is comparable."""
    if not name:
        return "?"
    for mj, cu in LINK_ALIASES.items():
        if name.startswith(mj):
            return cu
    if name.startswith("panda_"):
        return name.split("__")[0]
    for i in range(8):
        if f"link{i}" in name:
            return f"panda_link{i}"
    if "finger" in name:
        return "panda_finger"
    if "hand" in name:
        return "panda_hand"
    return name


def set_arm_q(dom, mon, q):
    """Drive the MuJoCo arm to a joint configuration and refresh derived quantities.

    `dist_engine.begin_frame()` is not optional. The labeler caches each geom's world-space hull
    for the duration of a frame, so without the clear every sample is measured against the FIRST
    configuration's geometry and the distance comes out identical at every point on the route --
    which is exactly what the first run of this script reported.
    """
    m, dd = dom.sim.model, dom.sim.data
    for i, v in enumerate(np.asarray(q, dtype=np.float64).ravel()[:7]):
        adr = m.get_joint_qpos_addr(f"robot0_joint{i + 1}")
        dd.qpos[adr] = float(v)
    dom.sim.forward()
    mon.dist_engine.begin_frame()


def labeler_distances(mon, dom, obstacle_hull_world, obstacle_geoms):
    """(union-hull distance, exact-piece distance, worst link) for the arm as MuJoCo hulls.

    Uses the labeler's own hull cache and the C++ GJK core, so the geometry and the arithmetic are
    the certifier's, not a reimplementation.
    """
    dd = dom.sim.data
    best_e = np.inf
    per_geom = []
    for g in mon.idx.robot_geoms:
        Av, Apad = mon.dist_engine._w(dd, g)
        du = max(0.0, fastgjk.gjk_distance(Av, obstacle_hull_world) - Apad)
        per_geom.append((du, mon.idx.geom_label(g), g))
    per_geom.sort(key=lambda x: x[0])
    best_u, worst_geom = per_geom[0][0], per_geom[0][1]
    link1 = canon_link(worst_geom)
    link2 = "-"
    for du, nm, _g in per_geom[1:]:
        if canon_link(nm) != link1:
            link2 = canon_link(nm)
            break
    for du, _nm, g in per_geom:
        if du > best_e:                       # union hull bounds every piece from below
            continue
        Av, Apad = mon.dist_engine._w(dd, g)
        for og in obstacle_geoms:
            Bv, Bpad = mon.dist_engine._w(dd, og)
            de = max(0.0, fastgjk.gjk_distance(Av, Bv) - Apad - Bpad)
            if de < best_e:
                best_e = de
    return float(best_u), float(best_e), link1, worst_geom, link2


def main():
    tdt = TensorDeviceType()
    robot_yml = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_yml, tdt)
    kc = robot_cfg.kinematics.kinematics_config
    joint_names = list(kc.joint_names)
    sph2link = kc.link_sphere_idx_map.detach().cpu().numpy()
    idx2link = {v: k for k, v in kc.link_name_to_idx_map.items()}

    base = dict(np.load(BASELINE, allow_pickle=True))
    rows, scenes = [], []

    for ii in INITS:
        key = f"{TAG}__i{ii:02d}__eef"
        if key not in base:
            print(f"[i{ii:02d}] no baseline trajectory, skipped", flush=True)
            continue
        env = None
        try:
            env, dom, mon, sg, task, init, obs = D.make_env(TAG, ii, res=84)
            D.set_render(dom, False)
            for _ in range(D.CONF["SETTLE_STEPS"]):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            obstacle = D.SPEC["names"][TAG]["obstacle"]
            info = D.eval_rule_shift(sg, obstacle, np.asarray(base[key]), INTRUSION_M)
            if not info.get("feasible") or info.get("skipped"):
                print(f"[i{ii:02d}] eval rule infeasible: {info.get('reason')}", flush=True)
                continue
            shift = np.asarray(info["shift"], float)
            import geom_shift
            _, _, anch = geom_shift.obstacle_anchors(dom)
            st = geom_shift.apply_shift(dom, anch, init[ii], shift)
            dom, mon, sg, obs, _ = D.setup_scene(env, st, ii, shift=shift, render=False)

            # the route the scripted planner failed on: the carried traverse, lift -> pre_place
            home = np.asarray(obs["robot0_eef_pos"], float).copy()
            hq = np.asarray(obs["robot0_eef_quat"], float).copy()
            wp = D.nominal_plan(sg, dom, TAG, home, hq)
            lift = next(w for w in wp if str(w[2]).startswith("lift:"))
            prep = next(w for w in wp if str(w[2]).startswith("pre_place:"))
            p0, p1, quat = np.asarray(lift[0], float), np.asarray(prep[0], float), lift[4]

            meshes = W.scene_meshes(dom, mon, sg, TAG, carried=None)
            base_p, base_R = W.robot_base_offset(dom)
            hull_w = mon._obstacle_hull_world(dom.sim.data, obstacle)
            ogeoms = list(mon.obstacle_geoms[obstacle])

            # The comparison world holds the OBSTACLE ONLY. With the table in it, the Panda's
            # base link sits inside the table box -- the robot is mounted at table height -- so
            # every query returned panda_link0 as the culprit and the comparison was meaningless.
            # The full world (table, receptacle) is what step 3 plans against; this is a
            # geometry-agreement check on the thing the demos have to avoid.
            obs_meshes = [d for d in meshes if d["name"].startswith("obstacle__")]
            rw = RobotWorld(RobotWorldConfig.load_from_config(
                robot_cfg,
                WorldConfig(mesh=[Mesh(name=d["name"], pose=[0, 0, 0, 1, 0, 0, 0],
                                       vertices=d["vertices"].tolist(),
                                       faces=d["faces"].tolist()) for d in obs_meshes]),
                tensor_args=tdt, collision_activation_distance=0.0,
                n_meshes=len(obs_meshes) + 4))
            ik = IKSolver(IKSolverConfig.load_from_robot_config(
                robot_cfg, None, rotation_threshold=0.08, position_threshold=0.008,
                num_seeds=24, self_collision_check=False, tensor_args=tdt))

            ts = np.linspace(0.0, 1.0, N_SAMPLES)
            line = np.asarray([p0 + t * (p1 - p0) for t in ts])
            line_b = W.to_base(line, base_p, base_R)
            quat_t = torch.tensor([[float(quat[0]), float(quat[1]), float(quat[2]),
                                    float(quat[3])]] * len(line_b), device=tdt.device)
            goal = Pose(torch.as_tensor(line_b, dtype=torch.float32, device=tdt.device), quat_t)
            res = ik.solve_batch(goal)
            qs = res.solution.detach().cpu().numpy().reshape(len(line_b), -1)
            ok = res.success.detach().cpu().numpy().ravel()

            n_solved = 0
            for k in range(len(line_b)):
                if not bool(ok[k]):
                    continue
                n_solved += 1
                q = qs[k][:7]
                qt = torch.as_tensor(q.reshape(1, -1).astype(np.float32), device=tdt.device)
                st_k = rw.get_kinematics(qt)
                sph = st_k.link_spheres_tensor.view(1, 1, -1, 4)
                # get_collision_distance SUMS over spheres (collision_cost.forward reduces),
                # so it answers "does this configuration collide" but not "where". The first
                # version of this script took argmax of that reduced tensor, got index 0 every
                # time, and reported panda_link0 -- the immovable base -- as the culprit on every
                # sample. For the per-link answer the spheres are queried one at a time.
                dsum = rw.get_collision_distance(sph).detach().cpu().numpy().ravel()
                pen = float(dsum.sum())
                per = np.zeros(sph.shape[2], dtype=np.float64)
                sph1 = sph.view(-1, 4)
                for si in range(sph1.shape[0]):
                    if float(sph1[si, 3]) <= 0.0:
                        continue
                    one = sph1[si].view(1, 1, 1, 4).clone()
                    per[si] = float(rw.get_collision_distance(one).detach().cpu().numpy().sum())
                order = np.argsort(-per)
                j = int(order[0])
                pen = float(per.max())
                plink = idx2link.get(int(sph2link[j]), "?") if per.max() > 0 else "-"
                # the runner-up link, because the two models partition the wrist differently and
                # "off by one link in the chain" is a different finding from "disagrees"
                plink2 = "-"
                for jj in order[1:]:
                    nm = idx2link.get(int(sph2link[int(jj)]), "?")
                    if per[int(jj)] > 0.0 and nm != plink:
                        plink2 = nm
                        break
                set_arm_q(dom, mon, q)
                du, de, llink, lgeom, llink2 = labeler_distances(mon, dom, hull_w, ogeoms)
                rows.append(dict(init=ii, k=k, t=round(float(ts[k]), 3),
                                 planner_penetration_m=round(pen, 5),
                                 planner_collides=bool(pen > 0.0), planner_link=plink,
                                 planner_link_2nd=plink2, labeler_geom=lgeom,
                                 labeler_link_2nd=llink2,
                                 labeler_union_m=round(du, 5), labeler_exact_m=round(de, 5),
                                 labeler_collides=bool(du <= 0.0), labeler_link=llink,
                                 agree_decision=bool((pen > 0.0) == (du <= 0.0)),
                                 agree_link=bool(plink == llink),
                                 agree_link_adjacent=bool(plink == llink or plink == llink2
                                                          or plink2 == llink)))
            print(f"[i{ii:02d}] shift {100*np.linalg.norm(shift):5.2f} cm   IK solved "
                  f"{n_solved}/{N_SAMPLES}   meshes {len(meshes)}", flush=True)
            scenes.append(dict(init=ii, shift_cm=round(100 * float(np.linalg.norm(shift)), 3),
                               n_meshes=len(meshes), n_obstacle_meshes=len(obs_meshes),
                               ik_solved=n_solved, n_samples=N_SAMPLES,
                               mesh_table=W.summarise(meshes)))
            np.savez_compressed(os.path.join(OUT, f"verify_world_i{ii:02d}.npz"),
                                line=line, line_base=line_b, q=qs, ik_ok=ok, shift=shift,
                                obstacle_hull_world=hull_w)
            del rw, ik
            torch.cuda.empty_cache()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    report(rows, scenes)


def report(rows, scenes):
    print()
    print("=" * 104)
    print("STEP 2 -- planner world vs the GJK labeler, same joint configurations")
    print("=" * 104)
    hdr = (f"  {'init':>4} {'t':>5} {'planner pen':>12} {'coll':>5} {'planner link':>14}  "
           f"{'labeler union':>14} {'exact':>8} {'coll':>5} {'labeler link':>14}  {'agree':>6}")
    print(hdr)
    for r in rows:
        print(f"  {r['init']:>4} {r['t']:>5.2f} {r['planner_penetration_m']:>12.5f} "
              f"{str(r['planner_collides']):>5} {r['planner_link']:>14}  "
              f"{r['labeler_union_m']:>14.5f} {r['labeler_exact_m']:>8.5f} "
              f"{str(r['labeler_collides']):>5} {r['labeler_link']:>14}  "
              f"{('OK' if r['agree_decision'] else 'DIFFER'):>6}")
    n = len(rows)
    if n:
        dec = sum(r["agree_decision"] for r in rows)
        coll_rows = [r for r in rows if r["labeler_collides"] or r["planner_collides"]]
        lnk = sum(r["agree_link"] for r in coll_rows)
        adj = sum(r.get("agree_link_adjacent", False) for r in coll_rows)
        print()
        print(f"  samples compared            {n}")
        print(f"  agree on collide/clear      {dec}/{n}")
        print(f"  colliding samples           {len(coll_rows)}")
        print(f"  of those, same link         {lnk}/{len(coll_rows)}")
        print(f"  same or neighbouring link   {adj}/{len(coll_rows)}")
        per_init = {}
        for r in rows:
            d = per_init.setdefault(r["init"], dict(n=0, pc=0, lc=0))
            d["n"] += 1
            d["pc"] += int(r["planner_collides"])
            d["lc"] += int(r["labeler_collides"])
        print()
        print("  straight-line route, per scene:")
        for ii, d in sorted(per_init.items()):
            print(f"    init {ii:>2}: planner says collides on {d['pc']}/{d['n']} samples, "
                  f"labeler on {d['lc']}/{d['n']}")
        links = {}
        for r in coll_rows:
            links[(r["planner_link"], r["labeler_link"])] = \
                links.get((r["planner_link"], r["labeler_link"]), 0) + 1
        print()
        print("  culprit link, planner vs labeler:")
        for (a, b), c in sorted(links.items(), key=lambda kv: -kv[1]):
            print(f"    {a:>16}  vs  {b:<16}  {c:>3} samples  "
                  f"{'MATCH' if a == b else 'differ'}")
        print()
        print("  the labeler's actual MuJoCo geom on colliding samples:")
        gm = {}
        for r in coll_rows:
            gm[r.get("labeler_geom", "?")] = gm.get(r.get("labeler_geom", "?"), 0) + 1
        for nm, c in sorted(gm.items(), key=lambda kv: -kv[1]):
            print(f"    {nm:<40} {c:>3} samples")
    json.dump(dict(rows=rows, scenes=scenes),
              open(os.path.join(OUT, "verify_world.json"), "w"), indent=1, default=str)
    print()
    print("VERIFY_WORLD_DONE")


if __name__ == "__main__":
    main()
