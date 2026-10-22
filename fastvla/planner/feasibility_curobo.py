"""Step 1 feasibility: can cuRobo plan a collision-free Panda joint path around a box?

Milestone 7a step 1. The question is narrow and the answer has to be a measurement, not an
import that succeeds: plan between two configurations with one box obstacle between them, check
the returned path against the obstacle independently, and report the time per plan.

Independent check matters twice over. cuRobo reports success from its own collision model, so
every returned path is re-tested here with my own sphere-to-box arithmetic (_aabb_sphere_distance),
not the planner's collision engine. And the obstacle's pose is derived from the straight-line path
rather than guessed, so "the planner returned a clear path" cannot be satisfied by an obstacle that
was never in the way -- which is exactly what the first version of this script did.

Author: Varun Chandra
Env: planner (torch 2.2.0+cu121, cuRobo v0.7.6). GPU 1 only.
"""

import json
import os
import time

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

from curobo.geom.types import Cuboid, WorldConfig  # noqa: E402  warp-cache
from curobo.types.base import TensorDeviceType  # noqa: E402  warp-cache
from curobo.types.robot import JointState, RobotConfig  # noqa: E402  warp-cache
from curobo.util_file import (  # noqa: E402  warp-cache
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (  # noqa: E402  warp-cache
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

N_PLANS = int(os.environ.get("N_PLANS", "20"))
BOX_DIMS = [0.08, 0.08, 0.30]  # a tall thin box, roughly the perfume bottle on its box
TABLE_POSE = [0.5, 0.0, -0.05, 1, 0, 0, 0]
TABLE_DIMS = [1.2, 1.2, 0.1]

# Two configurations that differ only in the base joint, mirrored, so the arm swings through a
# 77 cm arc and there is room for an obstacle in the middle that neither endpoint touches.
Q_START = [-1.1, -0.4, 0.0, -2.2, 0.0, 1.8, 0.0]
Q_GOAL = [1.1, -0.4, 0.0, -2.2, 0.0, 1.8, 0.0]

# The box is NOT placed by hand. The first version of this script put it at a guessed pose, the
# straight joint-space line turned out to clear it by 5 cm, and the "collision-free path" the
# planner returned was therefore no evidence of avoidance at all -- the obstacle was not in the
# way. The box pose is now DERIVED from the straight-line path (see place_box_on_path), so the
# lerp provably collides and the planner has something to avoid.


def main():
    tdt = TensorDeviceType()
    print(f"device {tdt.device}  torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    robot_yml = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_yml, tdt)
    kc = robot_cfg.kinematics.kinematics_config
    joint_names = list(kc.joint_names)
    coll_links = list(kc.link_name_to_idx_map.keys())
    print(
        f"robot franka.yml  dof {len(joint_names)}  collision links {len(coll_links)}: "
        f"{', '.join(coll_links)}",
        flush=True,
    )

    # Pass 1: a table-only world, used purely to run forward kinematics along the straight line
    # so the box can be placed ON that line.
    world_fk = WorldConfig(cuboid=[Cuboid(name="table", pose=TABLE_POSE, dims=TABLE_DIMS)])
    mg_fk = MotionGen(MotionGenConfig.load_from_robot_config(robot_cfg, world_fk, tensor_args=tdt))
    lerp = np.linspace(np.asarray(Q_START), np.asarray(Q_GOAL), 64)
    box_pose, host_link = place_box_on_path(mg_fk, tdt, lerp, kc)
    print(
        f"box placed on the straight-line path at "
        f"[{box_pose[0]:.3f} {box_pose[1]:.3f} {box_pose[2]:.3f}] "
        f"(derived from the swept position of {host_link}, not chosen by hand)",
        flush=True,
    )
    del mg_fk
    torch.cuda.empty_cache()

    world = WorldConfig(
        cuboid=[
            Cuboid(name="obstacle_box", pose=box_pose, dims=BOX_DIMS),
            Cuboid(name="table", pose=TABLE_POSE, dims=TABLE_DIMS),
        ]
    )

    t0 = time.perf_counter()
    cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world,
        tensor_args=tdt,
        interpolation_dt=0.02,
        collision_activation_distance=0.02,
        num_trajopt_seeds=4,
        num_graph_seeds=4,
    )
    mg = MotionGen(cfg)
    mg.warmup(enable_graph=True)
    t_warm = time.perf_counter() - t0
    print(f"MotionGen built and warmed in {t_warm:.1f} s", flush=True)

    q_start = torch.tensor([Q_START], dtype=torch.float32, device=tdt.device)
    q_goal = torch.tensor([Q_GOAL], dtype=torch.float32, device=tdt.device)
    start_state = JointState.from_position(q_start, joint_names=joint_names)
    goal_state = JointState.from_position(q_goal, joint_names=joint_names)

    # WELL-POSEDNESS. Three conditions, asserted rather than hoped for, because two earlier
    # versions of this test were vacuous in two different ways: first the box was clear of the
    # straight line (so avoiding it proved nothing), then it overlapped the GOAL configuration
    # (so no plan could exist and 0/20 failures looked like a planner problem).
    d_start = world_min_distance(mg, robot_cfg, tdt, np.asarray([Q_START]), box_pose)
    d_goal = world_min_distance(mg, robot_cfg, tdt, np.asarray([Q_GOAL]), box_pose)
    d_line = world_min_distance(mg, robot_cfg, tdt, lerp, box_pose)
    print(
        f"well-posedness:  start {d_start:+.4f} m   goal {d_goal:+.4f} m   "
        f"straight line {d_line:+.4f} m",
        flush=True,
    )
    assert d_start > 0.02, f"start configuration is not clear of the box ({d_start:+.4f} m)"
    assert d_goal > 0.02, f"goal configuration is not clear of the box ({d_goal:+.4f} m)"
    assert d_line <= 0.0, (
        f"the straight line does not collide ({d_line:+.4f} m) -- nothing to avoid"
    )
    print(
        "  -> both endpoints clear, the straight line collides: there is something to avoid",
        flush=True,
    )

    times, results = [], []
    for i in range(N_PLANS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = mg.plan_single(
            start_state,
            mg.compute_kinematics(goal_state).ee_pose.clone(),
            MotionGenPlanConfig(max_attempts=4, enable_graph=True, enable_opt=True, timeout=10.0),
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        times.append(dt)
        ok = bool(r.success.item())
        n = int(r.get_interpolated_plan().position.shape[0]) if ok else 0
        dmin = None
        if ok:
            q = r.get_interpolated_plan().position.cpu().numpy()
            dmin = world_min_distance(mg, robot_cfg, tdt, q, box_pose)
            np.savez_compressed(
                os.path.join(OUT, f"curobo_plan_{i:02d}.npz"), q=q, dt=dt, min_dist=dmin
            )
        results.append(
            dict(
                i=i,
                success=ok,
                seconds=round(dt, 4),
                waypoints=n,
                min_dist_m=(None if dmin is None else round(float(dmin), 5)),
            )
        )
        print(
            f"  plan {i:02d}: success={ok} {dt * 1000:7.1f} ms  {n:4d} waypoints  "
            f"min dist {('--' if dmin is None else f'{dmin:+.4f} m')}",
            flush=True,
        )

    t = np.asarray(times)
    nok = sum(1 for r in results if r["success"])
    clear = [r["min_dist_m"] for r in results if r["success"]]
    summary = dict(
        planner="curobo",
        version="v0.7.6",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        n_plans=N_PLANS,
        n_success=nok,
        box_pose=[round(float(x), 5) for x in box_pose],
        box_dims=BOX_DIMS,
        start_min_dist_m=round(float(d_start), 5),
        goal_min_dist_m=round(float(d_goal), 5),
        straight_line_min_dist_m=round(float(d_line), 5),
        seconds_mean=round(float(t.mean()), 4),
        seconds_median=round(float(np.median(t)), 4),
        seconds_min=round(float(t.min()), 4),
        seconds_max=round(float(t.max()), 4),
        warmup_seconds=round(t_warm, 1),
        path_min_dist_min_m=(round(float(min(clear)), 5) if clear else None),
        rows=results,
    )
    with open(os.path.join(OUT, "feasibility_curobo.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print()
    print(
        f"SUCCESS {nok}/{N_PLANS}   mean {t.mean() * 1000:.1f} ms   median "
        f"{np.median(t) * 1000:.1f} ms   range {t.min() * 1000:.1f}-{t.max() * 1000:.1f} ms"
    )
    if clear:
        print(
            f"returned paths: worst min distance to world {min(clear):+.4f} m "
            f"({'ALL CLEAR' if min(clear) > 0 else 'A PATH COLLIDES'})"
        )
    print("FEASIBILITY_CUROBO_DONE")


def place_box_on_path(mg, tdt, lerp, kc):
    """Put the box where the arm actually goes, so the straight line cannot dodge it.

    Centres the box on the end-effector position at the midpoint of the straight joint-space
    line. Returns the pose and a note on how it was chosen.
    """
    q = torch.as_tensor(np.asarray(lerp, dtype=np.float32), device=tdt.device)
    mid = q.shape[0] // 2
    c = mg.kinematics.get_state(q[mid : mid + 1]).ee_position.detach().cpu().numpy()[0]
    return (
        [float(c[0]), float(c[1]), float(c[2]), 1.0, 0.0, 0.0, 0.0],
        "the end-effector position at the midpoint of the straight line",
    )


def _aabb_sphere_distance(centres, radii, box_pose, box_dims):
    """Signed distance from each sphere to an axis-aligned box. Positive is clear.

    Deliberately my own arithmetic rather than cuRobo's collision engine: the point of the check
    is to be independent of the thing that reported success. Both boxes here have identity
    orientation, so the exact point-to-box distance is the norm of the per-axis overshoot, and
    the inside case is the negative of the smallest distance to a face.
    """
    c = np.asarray(centres, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    half = 0.5 * np.asarray(box_dims, dtype=np.float64)
    d = np.abs(c - np.asarray(box_pose[:3], dtype=np.float64)) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(d.max(axis=-1), 0.0)
    return (outside + inside) - r


def world_min_distance(mg, robot_cfg, tdt, q_np, box_pose):
    """Minimum signed distance from the robot's collision spheres to the world over a joint path.

    Forward kinematics come from cuRobo (there is no second kinematics implementation to hand and
    writing one would test the wrong thing), but the distance arithmetic is the function above,
    not the planner's collision query. Positive means clear of every obstacle.
    """
    q = torch.as_tensor(np.asarray(q_np, dtype=np.float32), device=tdt.device)
    boxes = [(box_pose, BOX_DIMS), (TABLE_POSE, TABLE_DIMS)]
    worst = None
    for i in range(q.shape[0]):
        st = mg.kinematics.get_state(q[i : i + 1])
        sph = st.link_spheres_tensor.view(-1, 4).detach().cpu().numpy()
        keep = sph[:, 3] > 0.0  # cuRobo pads unused sphere slots with radius <= 0
        centres, radii = sph[keep, :3], sph[keep, 3]
        for pose, dims in boxes:
            d = float(_aabb_sphere_distance(centres, radii, pose, dims).min())
            worst = d if worst is None else min(worst, d)
    return worst


if __name__ == "__main__":
    main()
