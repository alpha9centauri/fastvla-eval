"""The hand-off candidates: world off, world on, world on + attached payload.

Standing rule from milestone 7a: before any hypothesis about the planner, run the standalone
solver with the world on and off and compare. Applied here to the 9 hand-off candidates (height
+8/+12/+16 cm x horizontal offset away from the bottle 0/3/6 cm), which all failed collision-aware
IK on the first attempt.

Three conditions per candidate, so the cause is attributable rather than guessed:

  OFF            kinematics only            -- is the pose reachable at all?
  ON             scene world, no payload    -- does the scene refuse it?
  ON + payload   scene world + attached bowl -- does the ATTACHED OBJECT refuse it?

The payload matters because cuRobo reported `sphere_fit: occupancy test failed, assuming cuboid
volume` when fitting the bowl, which is a coarse over-approximation: an oversized payload welded
to the gripper would refuse poses that are actually fine.

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
from fastvla.planner import route as R                   # noqa: E402

import torch                                            # noqa: E402
from curobo.geom.types import Mesh, WorldConfig          # noqa: E402
from curobo.types.base import TensorDeviceType           # noqa: E402
from curobo.types.math import Pose                       # noqa: E402
from curobo.types.robot import RobotConfig               # noqa: E402
from curobo.util_file import get_robot_configs_path, join_path, load_yaml   # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig          # noqa: E402

TAG, INIT = "oa1", int(os.environ.get("INIT", "0"))
BASELINE = os.path.join(DISP, "out", "baseline_17500.npz")


def world_cfg(meshes):
    return WorldConfig(mesh=[Mesh(name=d["name"], pose=[0, 0, 0, 1, 0, 0, 0],
                                  vertices=d["vertices"].tolist(),
                                  faces=d["faces"].tolist()) for d in meshes])


def main():
    tdt = TensorDeviceType()
    rc = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"], tdt)
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
    lift = next(w for w in wp if str(w[2]).startswith("lift:"))
    place = next(w for w in wp if str(w[2]).startswith("place:"))
    carried = str(lift[2]).split(":", 1)[1]
    base_p, base_R = W.robot_base_offset(dom)
    meshes = W.scene_meshes(dom, mon, sg, TAG, carried=carried)
    hull_w = mon._obstacle_hull_world(dom.sim.data, obstacle)

    router = R.CuroboRouter.get()
    eef_now = np.asarray(
        dom.sim.data.site_xpos[dom.sim.model.site_name2id("gripper0_grip_site")], float)
    router.calibrate(dom, eef_now, base_p, base_R)
    router.update_for_scene(meshes)
    quat = R.xyzw_to_wxyz(np.asarray(lift[4], float))

    # start configuration at the lift, for the attachment
    b0 = router.site_to_hand(W.to_base(np.asarray(lift[0], float).reshape(1, 3),
                                       base_p, base_R)[0], quat)
    q0, ok0 = router.ik(b0, quat)
    payload = W.carried_mesh_abs(dom, mon, carried)
    pv = payload["vertices"]
    print("=" * 104)
    print(f"HAND-OFF CANDIDATES -- {TAG} init {INIT}, shift {100 * np.linalg.norm(shift):.2f} cm")
    print("=" * 104)
    print(f"  payload hull: {len(pv)} verts, extent "
          f"{np.round(pv.max(axis=0) - pv.min(axis=0), 4)} m  (bowl is ~0.107 x 0.107 x 0.051)")
    print(f"  start IK at lift: {ok0}")

    ik_off = IKSolver(IKSolverConfig.load_from_robot_config(
        rc, None, rotation_threshold=0.05, position_threshold=0.01, num_seeds=32,
        self_collision_check=True, tensor_args=tdt))
    ik_on = IKSolver(IKSolverConfig.load_from_robot_config(
        rc, world_cfg(meshes), rotation_threshold=0.05, position_threshold=0.01, num_seeds=32,
        self_collision_check=True, tensor_args=tdt,
        collision_activation_distance=R.ACTIVATION_M,
        collision_cache={"obb": 8, "mesh": 64}))

    attached = router.attach_payload(q0, payload) if (payload is not None and ok0) else False
    print(f"  payload attached to the router's MotionGen: {attached}")
    print()
    print(f"  {'up cm':>6} {'out cm':>7} {'OFF':>6} {'ON':>6} {'ON+payload':>11}  "
          f"{'obstacle gap cm':>16}")

    p_place = np.asarray(place[0], float)
    centroid = hull_w.mean(axis=0)
    away = p_place[:2] - centroid[:2]
    away = away / max(1e-9, float(np.linalg.norm(away)))
    rows = []
    for up in R.HANDOFF_UP_M:
        for out in R.HANDOFF_OUT_M:
            cand_w = p_place + np.array([away[0] * out, away[1] * out, up])
            cand_b = router.site_to_hand(
                W.to_base(cand_w.reshape(1, 3), base_p, base_R)[0], quat)
            p = torch.as_tensor(cand_b.reshape(1, 3).astype(np.float32), device=tdt.device)
            qq = torch.as_tensor(quat.reshape(1, 4).astype(np.float32), device=tdt.device)
            r_off = bool(ik_off.solve_single(Pose(p, qq)).success.item())
            r_on = bool(ik_on.solve_single(Pose(p, qq)).success.item())
            _q, r_pay = router.ik_collision_free(cand_b, quat)
            gap = float(np.min(np.linalg.norm(hull_w - cand_w[None, :], axis=1)))
            print(f"  {100 * up:>6.0f} {100 * out:>7.0f} {str(r_off):>6} {str(r_on):>6} "
                  f"{str(r_pay):>11}  {100 * gap:>16.2f}")
            rows.append(dict(up_cm=100 * up, out_cm=100 * out, ik_off=r_off, ik_on=r_on,
                             ik_on_payload=r_pay, obstacle_gap_cm=round(100 * gap, 2)))
    router.detach_payload()
    env.close()

    n_off = sum(r["ik_off"] for r in rows)
    n_on = sum(r["ik_on"] for r in rows)
    n_pay = sum(r["ik_on_payload"] for r in rows)
    print()
    print(f"  reachable (world off)        {n_off}/{len(rows)}")
    print(f"  accepted with the scene      {n_on}/{len(rows)}")
    print(f"  accepted with scene+payload  {n_pay}/{len(rows)}")
    if n_off and not n_on:
        print("  -> the SCENE refuses every candidate: the hand-off region is blocked, not the arm")
    elif n_on and not n_pay:
        print("  -> the ATTACHED PAYLOAD is what refuses them; the scene alone accepts "
              f"{n_on}/{len(rows)}")
    elif n_pay:
        print("  -> at least one candidate is usable; the lowest accepted one is the hand-off")
    else:
        print("  -> not reachable even kinematically: the candidate region itself is wrong")
    json.dump(rows, open(os.path.join(OUT, "diag_handoff.json"), "w"), indent=1)
    print("DIAG_HANDOFF_DONE")


if __name__ == "__main__":
    main()
