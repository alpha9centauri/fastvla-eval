"""Is the planner/labeler link disagreement a geometry error or a naming partition?

Step 2 reported that on colliding samples the planner names `panda_link7` where the labeler names
`robot0_link6_collision`. Those are two different answers to "which link hits", and the difference
matters: if the two models place the same metal in different places the world model is wrong, but
if they agree on where the contact is and merely attribute it to adjacent links in their own
naming, the world model is fine and only the label differs.

The test: for each colliding sample, compare the WORLD POSITION of the planner's deepest sphere
with the closest point on the labeler's worst geom. If those are centimetres apart, it is a
partition difference; if they are tens of centimetres apart, it is a frame bug.

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
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (
    DISP,
    os.path.join(DISP, "datagen"),
    "/data_new/vchandr2",
    "/data_new/vchandr2/fastvla-eval",
    "/data_new/vchandr2/vla-safety-interp/src/physics",
    "/data_new/vchandr2/fastvla-eval/fastvla/labeler",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D  # noqa: E402
import fastgjk  # noqa: E402
import geom_shift  # noqa: E402
import torch  # noqa: E402
from curobo.geom.types import Mesh, WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # noqa: E402
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig  # noqa: E402

from fastvla.planner import world as W  # noqa: E402

TAG = "oa1"
INIT = int(os.environ.get("INIT", "0"))
INTRUSION_M = float(os.environ.get("INTRUSION_M", "0.05"))
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")


def main():
    z = dict(np.load(os.path.join(OUT, f"verify_world_i{INIT:02d}.npz"), allow_pickle=True))
    qs, ok = z["q"], z["ik_ok"]
    tdt = TensorDeviceType()
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], tdt
    )
    kc = robot_cfg.kinematics.kinematics_config
    sph2link = kc.link_sphere_idx_map.detach().cpu().numpy()
    idx2link = {v: k for k, v in kc.link_name_to_idx_map.items()}

    env, dom, mon, sg, task, init, obs = D.make_env(TAG, INIT, res=84)
    D.set_render(dom, False)
    for _ in range(D.CONF["SETTLE_STEPS"]):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    shift = np.asarray(z["shift"], float)
    _, _, anch = geom_shift.obstacle_anchors(dom)
    st = geom_shift.apply_shift(dom, anch, init[INIT], shift)
    dom, mon, sg, obs, _ = D.setup_scene(env, st, INIT, shift=shift, render=False)

    obstacle = D.SPEC["names"][TAG]["obstacle"]
    meshes = [d for d in W.scene_meshes(dom, mon, sg, TAG) if d["name"].startswith("obstacle__")]
    base_p, base_R = W.robot_base_offset(dom)
    rw = RobotWorld(
        RobotWorldConfig.load_from_config(
            robot_cfg,
            WorldConfig(
                mesh=[
                    Mesh(
                        name=d["name"],
                        pose=[0, 0, 0, 1, 0, 0, 0],
                        vertices=d["vertices"].tolist(),
                        faces=d["faces"].tolist(),
                    )
                    for d in meshes
                ]
            ),
            tensor_args=tdt,
            collision_activation_distance=0.0,
            n_meshes=len(meshes) + 4,
        )
    )

    print(f"init {INIT}, shift {100 * np.linalg.norm(shift):.2f} cm")
    print(
        f"  {'k':>3} {'planner link':>14} {'sphere world xyz':>26} "
        f"{'labeler geom':>26} {'geom closest xyz':>26} {'gap cm':>8} "
        f"{'radius':>8} {'gap-r cm':>9}"
    )
    rows = []
    for k in range(len(qs)):
        if not bool(ok[k]):
            continue
        q = qs[k][:7]
        qt = torch.as_tensor(q.reshape(1, -1).astype(np.float32), device=tdt.device)
        sph = rw.get_kinematics(qt).link_spheres_tensor.view(-1, 4)
        per = np.zeros(sph.shape[0])
        for si in range(sph.shape[0]):
            if float(sph[si, 3]) <= 0.0:
                continue
            per[si] = float(
                rw.get_collision_distance(sph[si].view(1, 1, 1, 4).clone())
                .detach()
                .cpu()
                .numpy()
                .sum()
            )
        if per.max() <= 0.0:
            continue
        j = int(np.argmax(per))
        c_base = sph[j, :3].detach().cpu().numpy().astype(np.float64)
        c_world = c_base @ base_R.T + base_p  # planner sphere centre, back in world

        W.robot_base_offset(dom)
        for i, v in enumerate(q):
            dom.sim.data.qpos[dom.sim.model.get_joint_qpos_addr(f"robot0_joint{i + 1}")] = float(v)
        dom.sim.forward()
        mon.dist_engine.begin_frame()
        hull_w = mon._obstacle_hull_world(dom.sim.data, obstacle)
        best = (np.inf, None, None)
        for g in mon.idx.robot_geoms:
            Av, Apad = mon.dist_engine._w(dom.sim.data, g)
            d = max(0.0, fastgjk.gjk_distance(Av, hull_w) - Apad)
            if d < best[0]:
                best = (d, mon.idx.geom_label(g), Av)
        _d, gname, Av = best
        # closest vertex of that geom's hull to the planner's sphere centre, as a position proxy
        gclose = Av[int(np.argmin(np.linalg.norm(Av - c_world[None, :], axis=1)))]
        gap = float(np.linalg.norm(gclose - c_world))
        # The sphere CENTRE is being compared with a point on the geom's SURFACE, so a gap of
        # about one sphere radius is what perfect agreement looks like. Subtracting the radius is
        # the only way to read this number; without it the first run of this check reported a
        # suspiciously constant 4.79 cm "offset" that was simply the wrist sphere's radius.
        radius = float(sph[j, 3])
        gap_surface = gap - radius
        plink = idx2link.get(int(sph2link[j]), "?")
        print(
            f"  {k:>3} {plink:>14} [{c_world[0]:+.3f} {c_world[1]:+.3f} {c_world[2]:+.3f}] "
            f"{gname:>26} [{gclose[0]:+.3f} {gclose[1]:+.3f} {gclose[2]:+.3f}] "
            f"{100 * gap:>8.2f} {100 * radius:>8.2f} {100 * gap_surface:>9.2f}"
        )
        rows.append(
            dict(
                k=k,
                planner_link=plink,
                labeler_geom=gname,
                sphere_radius_cm=round(100 * radius, 2),
                gap_minus_radius_cm=round(100 * gap_surface, 2),
                sphere_world=[round(float(x), 4) for x in c_world],
                geom_closest=[round(float(x), 4) for x in gclose],
                gap_cm=round(100 * gap, 2),
            )
        )
    env.close()
    if rows:
        g = np.asarray([r["gap_cm"] for r in rows])
        gr = np.asarray([r["gap_minus_radius_cm"] for r in rows])
        rr = np.asarray([r["sphere_radius_cm"] for r in rows])
        print()
        print(
            f"  centre-to-surface gap:  median {np.median(g):.2f} cm  "
            f"range {g.min():.2f}-{g.max():.2f}"
        )
        print(
            f"  sphere radius:          median {np.median(rr):.2f} cm  "
            f"range {rr.min():.2f}-{rr.max():.2f}"
        )
        print(
            f"  gap minus radius:       median {np.median(gr):.2f} cm  "
            f"range {gr.min():.2f}-{gr.max():.2f}"
        )
        print("  -> near zero means the two models put the contact in the same place and only")
        print("     attribute it to adjacent links; a large value would be a frame bug.")
    with open(os.path.join(OUT, f"link_offset_i{INIT:02d}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    print("CHECK_LINK_OFFSET_DONE")


if __name__ == "__main__":
    main()
