"""Step 3: a motion planner in place of dgcore's route stage, and nothing else.

Milestone 7a step 3. `dgcore.generate_demo` calls the route planner exactly once:

    wp, detours, shape = plan_detours(sg, obstacle, wp0, ceiling=ceil)

so that one call is the whole integration surface. This module provides a replacement with the
same signature and `install()` binds it over `dgcore.plan_detours` at runtime. **dgcore.py is not
edited and nothing is copied out of it.** Everything downstream -- the OSC executor, the 2 cm GJK
certification, the rendered-replay gate, the LeRobot writer, the TFDS converter -- runs unchanged,
on waypoints of exactly the shape it already consumes.

What is replaced and what is not:

  kept    the grasp pose, the pre-grasp, the lift, the pre-place, the place, the release and the
          retreat, all from dgcore's own primitives via `nominal_plan`
  planned the carried traverse only: lift -> pre_place, the segment where the bowl is in the
          gripper and the obstacle is in the way. This is the segment the scripted planner bent
          with a lateral arc or a lift-over and could not certify.

The joint path is converted back to end-effector waypoints by forward kinematics, because that is
what the OSC executor takes. The executor tracks position with a held reference orientation, so
each waypoint carries its own quaternion from the plan.

PLANNER=script keeps dgcore's own planner and installs nothing.

Author: Varun Chandra
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (_DISP, os.path.join(_DISP, "datagen"), "/data_new/vchandr2",
           "/data_new/vchandr2/fastvla-eval",
           "/data_new/vchandr2/vla-safety-interp/src/physics"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastvla.planner import world as W   # noqa: E402

# Tuned from step 1 and step 2, not guessed:
#  - step 1 measured returned paths sitting 7-18 mm from obstacles at activation 0.02, short of
#    the 2 cm certification floor;
#  - step 2 measured the planner calling contact 0.7-3.8 cm before the labeler does.
# Both say the activation distance has to exceed the certification floor, so it starts at 0.04.
ACTIVATION_M = float(os.environ.get("PLANNER_ACTIVATION_M", "0.04"))
MAX_ATTEMPTS = int(os.environ.get("PLANNER_MAX_ATTEMPTS", "4"))
PLAN_TIMEOUT_S = float(os.environ.get("PLANNER_TIMEOUT_S", "8.0"))
TILT_LIMIT_DEG = float(os.environ.get("PLANNER_TILT_LIMIT_DEG", "20.0"))
IK_POS_TOL_M = float(os.environ.get("PLANNER_IK_POS_TOL_M", "0.01"))
WAYPOINT_STRIDE = int(os.environ.get("PLANNER_WAYPOINT_STRIDE", "3"))
MESH_CACHE = int(os.environ.get("PLANNER_MESH_CACHE", "64"))
# The planner hands the route back to dgcore's primitives ABOVE the place pose instead of planning
# all the way into it. Measured reason: on an eval-rule oa1 scene the pre_place pose sits 2.35 cm
# from the displaced bottle -- the obstacle is in the bowl's descent to the plate, which is the
# whole point of that placement -- so any activation distance above 2.35 cm makes the goal itself
# "in collision" and every plan fails with IK_FAIL. The terminal approach is dgcore's to own
# (grasp and place poses come from the existing primitives) and the certifier judges it, so the
# planner aims at a hand-off point this far above pre_place and stops there.
HANDOFF_UP_M = [float(x) for x in
                os.environ.get("PLANNER_HANDOFF_UP_M", "0.06,0.10,0.14,0.0").split(",")]
# Below the certification floor there is no route for anyone: that is a scene rejection, not a
# planner failure, and dgcore already has a name for it.
CERTIFY_FLOOR_M = 0.02

_STATS = []          # per-plan timings and outcomes, read by the step 4 driver


def stats():
    return list(_STATS)


def reset_stats():
    _STATS.clear()


class CuroboRouter:
    """One MotionGen for the whole process; the world is swapped per scene.

    Step 1 measured 11.3 s of build and warmup against ~0.2 s per plan, so building this per demo
    would dominate the run. The object is created once, cached on the module, and
    `update_for_scene` replaces the obstacle set.
    """

    _singleton = None

    @classmethod
    def get(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.geom.types import WorldConfig

        self.torch = torch
        self.tdt = TensorDeviceType()
        yml = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(yml, self.tdt)
        kc = self.robot_cfg.kinematics.kinematics_config
        self.joint_names = list(kc.joint_names)
        t0 = time.perf_counter()
        # The collision cache has to be sized up front. Building with an empty WorldConfig and
        # filling it later through update_world raises "Primitive Collision has no obstacles",
        # because the cache was allocated for zero meshes. MESH_CACHE is comfortably above the 14
        # meshes an eval-rule oa1 scene produces.
        cfg = MotionGenConfig.load_from_robot_config(
            self.robot_cfg, WorldConfig(), tensor_args=self.tdt,
            interpolation_dt=0.02, collision_activation_distance=ACTIVATION_M,
            num_trajopt_seeds=4, num_graph_seeds=4, collision_checker_type=_mesh_checker(),
            collision_cache={"obb": 8, "mesh": MESH_CACHE},
        )
        self.mg = MotionGen(cfg)
        self.mg.warmup(enable_graph=True)
        # A DEDICATED IK solver, not MotionGen.solve_ik. After warmup(enable_graph=True)
        # MotionGen's internal IK has CUDA graphs captured for the planning solve shape, and an
        # ad-hoc solve_ik call with a different goal shape fails with "changing goal type, cuda
        # graph reset not available" -- which cuRobo reports as MotionGenStatus.IK_FAIL. That
        # looked exactly like an unreachable goal and cost most of step 3's debugging: the same
        # poses solve to 0.0 mm through a standalone IKSolver, which is what step 2 used.
        # Kinematics only, no world: collisions along the route are plan_single's job, and the
        # terminal approach is dgcore's and the certifier's.
        self.ik_solver = IKSolver(IKSolverConfig.load_from_robot_config(
            self.robot_cfg, None, rotation_threshold=0.05, position_threshold=IK_POS_TOL_M,
            num_seeds=32, self_collision_check=True, tensor_args=self.tdt))
        self.warmup_s = time.perf_counter() - t0
        self.tool_offset = None

    def update_for_scene(self, meshes):
        from curobo.geom.types import Mesh, WorldConfig
        wc = WorldConfig(mesh=[Mesh(name=d["name"], pose=[0, 0, 0, 1, 0, 0, 0],
                                    vertices=d["vertices"].tolist(),
                                    faces=d["faces"].tolist()) for d in meshes])
        self.mg.update_world(wc)

    def ik(self, pos_base, quat):
        """One IK solve; returns (q, ok). Seeded from the retract config by cuRobo itself."""
        from curobo.types.math import Pose
        t = self.torch
        p = t.as_tensor(np.asarray(pos_base, dtype=np.float32).reshape(1, 3), device=self.tdt.device)
        q = t.as_tensor(np.asarray(quat, dtype=np.float32).reshape(1, 4), device=self.tdt.device)
        res = self.ik_solver.solve_single(Pose(p, q))
        ok = bool(res.success.item())
        sol = res.solution.detach().cpu().numpy().reshape(-1)[:7] if ok else None
        return sol, ok

    def plan(self, q_start, q_goal):
        """Plan between two joint configurations. Returns (joint path, seconds, reason).

        The goal is handed over as a cuRobo-built Pose (compute_kinematics(...).ee_pose.clone()),
        not a Pose assembled from numpy. plan_single runs under captured CUDA graphs after
        warmup(enable_graph=True) and rejects a goal whose buffer shape differs from the warmed
        one, reporting it as MotionGenStatus.IK_FAIL. Step 1's working feasibility call built the
        goal this way; building it from numpy here is what made every plan fail.
        """
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        t = self.torch
        st = JointState.from_position(
            t.as_tensor(np.asarray(q_start, dtype=np.float32).reshape(1, -1),
                        device=self.tdt.device),
            joint_names=self.joint_names)
        gs = JointState.from_position(
            t.as_tensor(np.asarray(q_goal, dtype=np.float32).reshape(1, -1),
                        device=self.tdt.device),
            joint_names=self.joint_names)
        goal = self.mg.compute_kinematics(gs).ee_pose.clone()
        t0 = time.perf_counter()
        r = self.mg.plan_single(st, goal, MotionGenPlanConfig(
            max_attempts=MAX_ATTEMPTS, enable_graph=True, enable_opt=True,
            timeout=PLAN_TIMEOUT_S))
        dt = time.perf_counter() - t0
        if not bool(r.success.item()):
            return None, dt, str(getattr(r, "status", "no_solution"))
        return r.get_interpolated_plan().position.detach().cpu().numpy(), dt, "ok"

    def calibrate(self, dom, eef_pos_world, base_p, base_R):
        """Measure the tool offset between robosuite's eef site and cuRobo's panda_hand frame.

        These are not the same frame. robosuite reports `robot0_eef_pos` at the gripper site,
        9.65 cm beyond the hand frame cuRobo's franka.yml treats as the end effector, so handing a
        site position to cuRobo as a goal asks the HAND to be where the fingertips should be. On
        oa1 that put every pre_place goal about 10 cm too far out and IK failed at the reach limit
        -- reported as MotionGenStatus.IK_FAIL, which reads like a crowded scene and is not.

        Measured here rather than hard-coded, from the scene's own joint configuration, so a
        different gripper calibrates itself instead of silently inheriting a Panda constant. The
        offset is stored in the HAND frame, where it is constant.
        """
        m, dd = dom.sim.model, dom.sim.data
        q = np.array([dd.qpos[m.get_joint_qpos_addr(f"robot0_joint{i + 1}")] for i in range(7)])
        fp, fq = self.fk(q.reshape(1, -1))
        site_b = (np.asarray(eef_pos_world, float) - base_p) @ base_R
        d_base = site_b - fp[0]
        R_h = _quat_to_mat(fq[0])
        self.tool_offset = R_h.T @ d_base
        self.tool_offset_norm = float(np.linalg.norm(self.tool_offset))
        return self.tool_offset

    def site_to_hand(self, pos_base, quat_wxyz):
        """A desired eef-site position -> the hand position cuRobo should be asked for."""
        if getattr(self, "tool_offset", None) is None:
            return np.asarray(pos_base, float)
        return np.asarray(pos_base, float) - _quat_to_mat(quat_wxyz) @ self.tool_offset

    def hand_to_site(self, pos_base, quat_wxyz):
        """The inverse, for turning a planned hand path back into eef-site waypoints."""
        if getattr(self, "tool_offset", None) is None:
            return np.asarray(pos_base, float)
        return np.asarray(pos_base, float) + _quat_to_mat(quat_wxyz) @ self.tool_offset

    def fk(self, q):
        """Joint path -> (positions in base frame, quaternions wxyz)."""
        t = self.torch
        qq = t.as_tensor(np.asarray(q, dtype=np.float32), device=self.tdt.device)
        st = self.mg.kinematics.get_state(qq)
        return (st.ee_position.detach().cpu().numpy().astype(np.float64),
                st.ee_quaternion.detach().cpu().numpy().astype(np.float64))


def _mesh_checker():
    from curobo.geom.sdf.world import CollisionCheckerType
    return CollisionCheckerType.MESH


def _quat_to_mat(q_wxyz):
    """wxyz quaternion -> 3x3 rotation, via MuJoCo so the convention matches the simulator."""
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=np.float64))
    return R.reshape(3, 3)


def xyzw_to_wxyz(q):
    """robosuite/MuJoCo report end-effector orientation as xyzw; cuRobo's Pose takes wxyz.

    Getting this wrong does not error, it silently asks for an unreachable orientation: the first
    integrated run failed every plan with MotionGenStatus.IK_FAIL on the pre_place goal in 0.3 s,
    which is what an impossible goal looks like rather than a crowded scene.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def wxyz_to_xyzw(q):
    """The inverse, for waypoints handed back to dgcore's executor."""
    q = np.asarray(q, dtype=np.float64).ravel()
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _quat_angle_deg(a, b):
    """Angle between two wxyz quaternions, in degrees."""
    a = np.asarray(a, float) / max(1e-12, np.linalg.norm(a))
    b = np.asarray(b, float) / max(1e-12, np.linalg.norm(b))
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))


def plan_detours_curobo(sg, obstacle, wp, shift=None, ceiling=None):
    """dgcore.plan_detours' signature, backed by cuRobo. Returns (waypoints, detours, shape).

    `shape` carries the keys dgcore's caller reads: plan_clears, and blocked_at_terminal when the
    scene itself is the problem rather than the route.
    """
    import dgcore as D
    dom, mon = sg.dom, sg.mon
    router = CuroboRouter.get()
    base_p, base_R = W.robot_base_offset(dom)

    # which object is carried on this leg, so it can be attached and removed from the obstacle set
    carried = None
    for w in wp:
        if str(w[2]).startswith("lift:"):
            carried = str(w[2]).split(":", 1)[1]
            break
    meshes = W.scene_meshes(dom, mon, sg, _tag_of(sg, D), carried=carried)
    router.update_for_scene(meshes)
    # Calibrate the tool offset against this scene's own arm before any goal is converted.
    eef_now = np.asarray(dom.sim.data.site_xpos[dom.sim.model.site_name2id("gripper0_grip_site")],
                         float)
    router.calibrate(dom, eef_now, base_p, base_R)

    i_lift = next((i for i, w in enumerate(wp) if str(w[2]).startswith("lift:")), None)
    i_pre = next((i for i, w in enumerate(wp) if str(w[2]).startswith("pre_place:")), None)
    if i_lift is None or i_pre is None or i_pre <= i_lift:
        return wp, [], dict(plan_clears=False, blocked_at_terminal=None,
                            reason="no carried traverse in the nominal plan")

    p0 = np.asarray(wp[i_lift][0], float)
    p1 = np.asarray(wp[i_pre][0], float)
    # Is the terminal pose itself inside the certification floor? Then no route exists and the
    # scene is the problem, which dgcore reports separately from a planner failure.
    hull_w = mon._obstacle_hull_world(dom.sim.data, obstacle)
    term_gap = float(np.min(np.linalg.norm(hull_w - p1[None, :], axis=1)))
    if term_gap < CERTIFY_FLOOR_M:
        _STATS.append(dict(stage="terminal", seconds=0.0, ok=False,
                           terminal_gap_m=round(term_gap, 4)))
        return wp, [], dict(plan_clears=False, blocked_at_terminal="pre_place",
                            terminal_gap_m=round(term_gap, 4),
                            reason=f"scene_reject:terminal_too_close:{100 * term_gap:.1f}cm")
    quat_xyzw = np.asarray(wp[i_lift][4], float)
    quat = xyzw_to_wxyz(quat_xyzw)          # cuRobo wants wxyz
    b0 = router.site_to_hand(W.to_base(p0.reshape(1, 3), base_p, base_R)[0], quat)
    b1 = router.site_to_hand(W.to_base(p1.reshape(1, 3), base_p, base_R)[0], quat)

    q0, ok = router.ik(b0, quat)
    if not ok:
        _STATS.append(dict(stage="ik_start", seconds=0.0, ok=False))
        return wp, [], dict(plan_clears=False, blocked_at_terminal=None,
                            reason="planner_failure:ik_lift")
    # Round-trip the start pose. Comparing cuRobo's IK against cuRobo's FK, as the first version
    # did, is blind to a frame error: it round-trips the wrong target consistently. The comparison
    # that matters is against the SITE position we actually wanted.
    fk_p, fk_q = router.fk(np.asarray(q0).reshape(1, -1))
    site_b = router.hand_to_site(fk_p[0], fk_q[0])
    want_b = W.to_base(p0.reshape(1, 3), base_p, base_R)[0]
    rt_pos = float(np.linalg.norm(site_b - want_b))
    rt_rot = _quat_angle_deg(quat, fk_q[0])
    if rt_pos > IK_POS_TOL_M or rt_rot > TILT_LIMIT_DEG:
        _STATS.append(dict(stage="ik_roundtrip", seconds=0.0, ok=False,
                           pos_err_m=round(rt_pos, 4), rot_err_deg=round(rt_rot, 2)))
        return wp, [], dict(plan_clears=False, blocked_at_terminal=None,
                            reason=f"planner_failure:ik_roundtrip_{1000 * rt_pos:.0f}mm_"
                                   f"{rt_rot:.0f}deg")

    # Aim above pre_place and let the primitives do the descent. The offsets are tried in order
    # and the first that both IKs and plans wins; 0.0 is last so the direct goal is still
    # attempted when there is room for it.
    path, dt, why, used_up = None, 0.0, "no_attempt", None
    for up in HANDOFF_UP_M:
        goal_b = b1 + np.array([0.0, 0.0, float(up)])
        q_goal, ik_ok = router.ik(goal_b, quat)
        if not ik_ok:
            why = f"ik_goal_up{100 * up:.0f}cm"
            continue
        path, dti, why = router.plan(q0, q_goal)
        dt += dti
        if path is not None:
            used_up = up
            break
    _STATS.append(dict(stage="plan", seconds=round(dt, 4), ok=path is not None, reason=why,
                       handoff_up_m=used_up))
    if path is None:
        return wp, [], dict(plan_clears=False, blocked_at_terminal=None,
                            reason=f"planner_failure:curobo:{why}")

    pos_b, quats = router.fk(path)
    # back to eef-site positions, which is what the executor drives
    pos_b = np.asarray([router.hand_to_site(pos_b[k], quats[k]) for k in range(len(pos_b))])
    pos_w = pos_b @ base_R.T + base_p[None, :]

    # The plan is free to re-orient the wrist; the carried bowl is not. Reject a path that tips the
    # grasp orientation beyond the limit rather than executing it and failing certification later.
    tilt = max(_quat_angle_deg(quat, qv) for qv in quats)
    if tilt > TILT_LIMIT_DEG:
        return wp, [], dict(plan_clears=False, blocked_at_terminal=None,
                            reason=f"planner_failure:tilt_{tilt:.1f}deg")

    # thin the interpolated path: the executor drives to each waypoint with a tolerance, so every
    # 20 ms sample would make the episode needlessly long
    keep = list(range(1, len(pos_w) - 1, max(1, WAYPOINT_STRIDE)))
    carry = wp[i_lift][3]
    via = [(pos_w[k], D.CLOSE, f"detour_curobo_v{n}", carry, wxyz_to_xyzw(quats[k]))
           for n, k in enumerate(keep)]

    out = list(wp[:i_lift + 1]) + via + list(wp[i_pre:])
    detours = [dict(kind="curobo_joint_path", n_via=len(via), seconds=round(dt, 4),
                    tilt_deg=round(tilt, 2), waypoints=int(len(pos_w)),
                    handoff_up_m=used_up, terminal_gap_m=round(term_gap, 4))]
    shape = dict(plan_clears=True, blocked_at_terminal=None, planner="curobo",
                 plan_seconds=round(dt, 4), n_via=len(via), tilt_deg=round(tilt, 2),
                 handoff_up_m=used_up, terminal_gap_m=round(term_gap, 4))
    return out, detours, shape


def _tag_of(sg, D):
    """Which task this scene is, by matching the declared obstacle against dgcore's spec."""
    for tag, spec in D.SPEC["names"].items():
        if spec.get("obstacle") in set(sg.obs_geoms):
            return tag
    return "oa1"


def install(planner=None):
    """Bind the chosen route planner over dgcore's. Returns the name actually installed."""
    import dgcore as D
    name = (planner or os.environ.get("PLANNER", "script")).lower()
    if name == "script":
        return "script"
    if name == "curobo":
        if not hasattr(D, "_plan_detours_script"):
            D._plan_detours_script = D.plan_detours
        D.plan_detours = plan_detours_curobo
        return "curobo"
    if name == "ompl":
        raise NotImplementedError(
            "OMPL was the step 1 fallback and was never needed: cuRobo builds and runs here "
            "(see FEASIBILITY.md). PLANNER=ompl is left unimplemented rather than stubbed.")
    raise ValueError(f"unknown PLANNER={name!r}; expected script, curobo or ompl")


def uninstall():
    import dgcore as D
    if hasattr(D, "_plan_detours_script"):
        D.plan_detours = D._plan_detours_script
