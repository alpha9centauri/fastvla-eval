"""Is the pre_place goal in collision in cuRobo's sphere model? World on vs world off.

The test that should have been run first, and is now the standing rule: before any hypothesis
about the planner, run the standalone IKSolver with the world ON and OFF and compare. Everything
else -- frames, quaternions, graph capture, update_world -- is downstream of knowing whether the
goal is geometrically acceptable to cuRobo at all.

Step 2 measured cuRobo's sphere model calling contact 0.7-3.8 cm before GJK does, and the
pre_place pose sits 2.35 cm from the displaced bottle. If collision-aware IK refuses that pose,
MotionGenStatus.IK_FAIL is correct behaviour and not a bug to chase.

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
for _p in (
    DISP,
    os.path.join(DISP, "datagen"),
    "/data_new/vchandr2",
    "/data_new/vchandr2/fastvla-eval",
    "/data_new/vchandr2/vla-safety-interp/src/physics",
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D  # noqa: E402
import geom_shift  # noqa: E402
import torch  # noqa: E402
from curobo.geom.types import Mesh, WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.robot import RobotConfig  # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402

from fastvla.planner import route as R  # noqa: E402
from fastvla.planner import world as W  # noqa: E402

TAG, INIT = "oa1", int(os.environ.get("INIT", "0"))
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")
ACTIVATIONS = [float(x) for x in os.environ.get("ACTIVATIONS", "0.04,0.02,0.01,0.0").split(",")]


def world_cfg(meshes):
    return WorldConfig(
        mesh=[
            Mesh(
                name=d["name"],
                pose=[0, 0, 0, 1, 0, 0, 0],
                vertices=d["vertices"].tolist(),
                faces=d["faces"].tolist(),
            )
            for d in meshes
        ]
    )


def main():
    tdt = TensorDeviceType()
    rc = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], tdt
    )

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

    home = np.asarray(obs["robot0_eef_pos"], float)
    hq = np.asarray(obs["robot0_eef_quat"], float)
    wp = D.nominal_plan(sg, dom, TAG, home, hq)
    prep = next(w for w in wp if str(w[2]).startswith("pre_place:"))
    base_p, base_R = W.robot_base_offset(dom)
    meshes = W.scene_meshes(dom, mon, sg, TAG, carried="akita_black_bowl_1")

    # the goal, through the same tool-offset calibration route.py uses
    router = R.CuroboRouter.get()
    eef_now = np.asarray(
        dom.sim.data.site_xpos[dom.sim.model.site_name2id("gripper0_grip_site")], float
    )
    router.calibrate(dom, eef_now, base_p, base_R)
    quat = R.xyzw_to_wxyz(np.asarray(prep[4], float))
    site_b = W.to_base(np.asarray(prep[0], float).reshape(1, 3), base_p, base_R)[0]
    hand_b = router.site_to_hand(site_b, quat)

    hull_w = mon._obstacle_hull_world(dom.sim.data, obstacle)
    gap = float(np.min(np.linalg.norm(hull_w - np.asarray(prep[0], float)[None, :], axis=1)))

    print("=" * 100)
    print(
        f"GOAL IK -- world ON vs OFF, {TAG} init {INIT}, shift {100 * np.linalg.norm(shift):.2f} cm"
    )
    print("=" * 100)
    print(f"  pre_place site (base) {np.round(site_b, 4)}   hand (base) {np.round(hand_b, 4)}")
    print(f"  nearest obstacle hull vertex to the pre_place EE point: {100 * gap:.2f} cm")
    print(f"  scene meshes: {len(meshes)}")
    print()
    print(f"  {'world':>22} {'activation m':>13} {'IK':>6} {'pos err mm':>11} {'rot err':>9}")

    rows = []
    cases = [("OFF (kinematics only)", None, 0.0)]
    cases += [(f"ON ({len(meshes)} meshes)", meshes, a) for a in ACTIVATIONS]
    for label, ms, act in cases:
        ik = IKSolver(
            IKSolverConfig.load_from_robot_config(
                rc,
                world_cfg(ms) if ms is not None else None,
                rotation_threshold=0.05,
                position_threshold=0.01,
                num_seeds=32,
                self_collision_check=True,
                tensor_args=tdt,
                collision_activation_distance=act,
                collision_cache={"obb": 8, "mesh": 64} if ms is not None else None,
            )
        )
        p = torch.as_tensor(hand_b.reshape(1, 3).astype(np.float32), device=tdt.device)
        q = torch.as_tensor(quat.reshape(1, 4).astype(np.float32), device=tdt.device)
        res = ik.solve_single(Pose(p, q))
        ok = bool(res.success.item())
        pe = (
            float(res.position_error.min().item())
            if res.position_error is not None
            else float("nan")
        )
        re_ = (
            float(res.rotation_error.min().item())
            if res.rotation_error is not None
            else float("nan")
        )
        print(f"  {label:>22} {act:>13.3f} {str(ok):>6} {1000 * pe:>11.2f} {re_:>9.4f}")
        rows.append(
            dict(
                world=label,
                activation_m=act,
                ik_ok=ok,
                pos_err_mm=round(1000 * pe, 3),
                rot_err=round(re_, 5),
            )
        )
        del ik
        torch.cuda.empty_cache()

    env.close()
    off = rows[0]["ik_ok"]
    on_any = any(r["ik_ok"] for r in rows[1:])
    print()
    print(f"  world OFF: IK {'succeeds' if off else 'fails'}")
    print(
        f"  world ON : IK succeeds at {sum(r['ik_ok'] for r in rows[1:])} of "
        f"{len(rows) - 1} activation distances"
    )
    if off and not on_any:
        print("  VERDICT: CONFIRMED -- the goal is reachable but in collision in cuRobo's sphere")
        print("           model at every activation distance. IK_FAIL is correct behaviour.")
    elif off and on_any:
        print("  VERDICT: PARTIAL -- the world refuses the goal only at some activation")
        print("           distances; the lowest that accepts it is usable.")
    else:
        print("  VERDICT: NOT CONFIRMED -- something else is refusing this pose.")
    with open(os.path.join(OUT, "diag_goal_ik.json"), "w") as fh:
        json.dump(
            dict(
                rows=rows,
                terminal_gap_cm=round(100 * gap, 3),
                hand_base=[round(float(x), 5) for x in hand_b],
            ),
            fh,
            indent=1,
        )
    print("DIAG_GOAL_IK_DONE")


if __name__ == "__main__":
    main()
