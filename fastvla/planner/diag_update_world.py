"""Does `update_world` leave cuRobo's mesh world wrong?

The one hypothesis under test, and nothing else. STEP3_STATUS.md records that
`MotionGen.plan_single` returns IK_FAIL on every goal at every collision activation distance from
0.04 down to 0.005 -- a margin that makes no difference at any value is the signature of geometry
that is wrong rather than tight. Step 1's working feasibility script built MotionGen with a
non-empty world at construction and never called `update_world`; `route.py` builds with an empty
WorldConfig plus a sized collision cache and calls `update_world` per scene.

So: load one eval-rule oa1 scene's real meshes two ways, query the SAME collision checker
MotionGen's planner consults, at the retract configuration, and compare.

  A  MotionGen(empty world, collision_cache) then update_world(meshes)   <- what route.py does
  B  MotionGen(meshes at construction)                                   <- what step 1 did

If A reports collisions at the retract configuration and B does not, or A's mesh poses sit at the
origin, the bug is confirmed. If the two agree, that is written down and the hypothesis is dead.

Author: Varun Chandra
Env: planner. GPU 1 only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
           "/data_new/vchandr2/fastvla-eval",
           "/data_new/vchandr2/vla-safety-interp/src/physics"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D                                      # noqa: E402
import geom_shift                                       # noqa: E402
from fastvla.planner import world as W                   # noqa: E402

import torch                                            # noqa: E402
from curobo.geom.sdf.world import CollisionQueryBuffer   # noqa: E402
from curobo.geom.types import Mesh, WorldConfig          # noqa: E402
from curobo.types.base import TensorDeviceType           # noqa: E402
from curobo.types.robot import RobotConfig               # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml   # noqa: E402
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig       # noqa: E402

TAG, INIT = "oa1", int(os.environ.get("INIT", "0"))
ACTIVATION_M = float(os.environ.get("PLANNER_ACTIVATION_M", "0.04"))
MESH_CACHE = 64
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")


def build_scene():
    base = dict(np.load(BASELINE, allow_pickle=True))
    env, dom, mon, sg, task, init, obs = D.make_env(TAG, INIT, res=84)
    D.set_render(dom, False)
    for _ in range(D.CONF["SETTLE_STEPS"]):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    obstacle = D.SPEC["names"][TAG]["obstacle"]
    info = D.eval_rule_shift(sg, obstacle, np.asarray(base[f"{TAG}__i{INIT:02d}__eef"]), 0.05)
    shift = np.asarray(info["shift"], float)
    _, _, anch = geom_shift.obstacle_anchors(dom)
    st = geom_shift.apply_shift(dom, anch, init[INIT], shift)
    dom, mon, sg, obs, _ = D.setup_scene(env, st, INIT, shift=shift, render=False)
    meshes = W.scene_meshes(dom, mon, sg, TAG, carried="akita_black_bowl_1")
    return env, dom, mon, sg, meshes, shift


def world_cfg(meshes):
    return WorldConfig(mesh=[Mesh(name=d["name"], pose=[0, 0, 0, 1, 0, 0, 0],
                                  vertices=d["vertices"].tolist(),
                                  faces=d["faces"].tolist()) for d in meshes])


def make_mg(meshes, mode):
    """mode 'update' = empty then update_world; mode 'direct' = meshes at construction."""
    tdt = TensorDeviceType()
    rc = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], tdt)
    from curobo.geom.sdf.world import CollisionCheckerType
    init_world = WorldConfig() if mode == "update" else world_cfg(meshes)
    cfg = MotionGenConfig.load_from_robot_config(
        rc, init_world, tensor_args=tdt, interpolation_dt=0.02,
        collision_activation_distance=ACTIVATION_M, num_trajopt_seeds=4, num_graph_seeds=4,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": 8, "mesh": MESH_CACHE})
    mg = MotionGen(cfg)
    mg.warmup(enable_graph=True)
    if mode == "update":
        mg.update_world(world_cfg(meshes))
    return mg, rc, tdt


def probe(mg, rc, tdt, label):
    """Per-sphere distance at the retract configuration, through MotionGen's own checker."""
    q = mg.get_retract_config().view(1, -1)
    st = mg.kinematics.get_state(q)
    sph = st.link_spheres_tensor.view(1, 1, -1, 4).contiguous()
    buf = CollisionQueryBuffer.initialize_from_shape(
        sph.shape, tdt, mg.world_collision.collision_types)
    d = mg.world_collision.get_sphere_distance(
        sph, buf, weight=tdt.to_device([1.0]), activation_distance=tdt.to_device([0.0]))
    d = d.detach().cpu().numpy().ravel()
    s = sph.view(-1, 4).detach().cpu().numpy()
    real = s[:, 3] > 0.0
    kc = rc.kinematics.kinematics_config
    s2l = kc.link_sphere_idx_map.detach().cpu().numpy()
    i2l = {v: k for k, v in kc.link_name_to_idx_map.items()}

    print(f"\n  --- {label} ---")
    print(f"  retract config: {np.round(q.detach().cpu().numpy().ravel(), 4)}")
    print(f"  spheres: {int(real.sum())} real of {len(s)}")
    hits = [(i, d[i]) for i in range(len(d)) if real[i] and d[i] > 0.0]
    print(f"  spheres reporting penetration (>0): {len(hits)}")
    for i, v in sorted(hits, key=lambda kv: -kv[1])[:10]:
        print(f"    sphere {i:>3} link {i2l.get(int(s2l[i]), '?'):>18}  "
              f"centre {np.round(s[i, :3], 3)} r {s[i, 3]:.3f}  penetration {v:+.5f}")
    print(f"  max penetration over all spheres: {float(d[real].max()):+.5f}")
    print(f"  sum over all spheres:             {float(d[real].sum()):+.5f}")

    wm = mg.world_model
    ms = getattr(wm, "mesh", None) or []
    print(f"  world_model holds {len(ms)} meshes")
    for mm in ms[:6]:
        pz = np.asarray(mm.pose, float)
        v = np.asarray(mm.vertices, float) if mm.vertices is not None else np.zeros((0, 3))
        bb = (np.round(v.min(axis=0), 3).tolist(), np.round(v.max(axis=0), 3).tolist()) \
            if len(v) else ("-", "-")
        print(f"    {mm.name:<34} pose {np.round(pz, 3)}  verts {len(v):>5}  bbox {bb[0]}..{bb[1]}")
    return dict(label=label, n_spheres=int(real.sum()), n_penetrating=len(hits),
                max_penetration=float(d[real].max()), sum_penetration=float(d[real].sum()),
                n_meshes=len(ms),
                mesh_poses=[[round(float(x), 4) for x in np.asarray(mm.pose, float)]
                            for mm in ms],
                mesh_names=[mm.name for mm in ms])


def main():
    env, dom, mon, sg, meshes, shift = build_scene()
    print("=" * 100)
    print(f"DIAG -- update_world vs direct construction, {TAG} init {INIT}, "
          f"shift {100 * np.linalg.norm(shift):.2f} cm, activation {ACTIVATION_M} m")
    print("=" * 100)
    print(f"  scene meshes from the labeler's hulls: {len(meshes)}")
    for r in W.summarise(meshes)[:4]:
        print(f"    {r['name']:<34} v{r['n_vertices']:>5}  bbox {r['bbox_min']}..{r['bbox_max']}")

    rows = []
    for mode, label in (("update", "A: MotionGen(empty) then update_world(meshes)"),
                        ("direct", "B: MotionGen(meshes at construction)")):
        mg, rc, tdt = make_mg(meshes, mode)
        rows.append(probe(mg, rc, tdt, label))
        del mg
        torch.cuda.empty_cache()
    env.close()

    a, b = rows
    print()
    print("=" * 100)
    same = (a["n_penetrating"] == b["n_penetrating"]
            and abs(a["max_penetration"] - b["max_penetration"]) < 1e-6)
    print(f"  A penetrating spheres {a['n_penetrating']}, max {a['max_penetration']:+.5f}, "
          f"meshes {a['n_meshes']}")
    print(f"  B penetrating spheres {b['n_penetrating']}, max {b['max_penetration']:+.5f}, "
          f"meshes {b['n_meshes']}")
    print(f"  VERDICT: {'SAME -- hypothesis dead' if same else 'DIFFERENT -- bug confirmed'}")
    json.dump(rows, open(os.path.join(OUT, "diag_update_world.json"), "w"), indent=1)
    print("DIAG_UPDATE_WORLD_DONE")


if __name__ == "__main__":
    main()
