"""Does cuRobo's attached-object fit actually cover the carried bowl?

Step 4 found the planner optimistic about the payload: paths it called collision-free measured
0.39 cm of GJK clearance at their tightest via. The suspect is the attachment, which step 3 set to
SAMPLE_SURFACE with 5 mm spheres after cuRobo's volume fit degenerated to a solid cuboid
("occupancy test failed, assuming cuboid volume") and refused every hand-off.

The measurement, as specified: sample 1,000 points uniformly on the bowl's TRUE convex hull
surface and, for each, find the distance to the nearest attached sphere's surface.

  gap(p) = min_i ( |p - c_i| - r_i )

  gap <= 0   the point lies inside some attached sphere -- covered
  gap  > 0   the hull sticks out of the sphere set by that much -- UNDER-COVERED

So the fit is adequate when max gap <= 0, which is the condition the brief asks for. (The brief's
parenthetical said "negative = under-covered"; with the natural signed-distance convention above it
is the positive values that mean under-coverage, and the "worst <= 0" target is the same either
way. Stated here so the sign cannot be misread.)

Fits and radii are swept and the numbers reported for each, so the choice is made on coverage
rather than on whether candidates happen to pass.

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
           "/data_new/vchandr2/fastvla-eval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dgcore as D                                  # noqa: E402
import geom_shift                                   # noqa: E402
from fastvla.planner import world as W               # noqa: E402
from fastvla.planner import route as R               # noqa: E402

import torch                                        # noqa: E402
from curobo.geom.sphere_fit import SphereFitType     # noqa: E402

TAG, INIT = "oa1", int(os.environ.get("INIT", "25"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "1000"))
BASELINE = os.path.join(DISP, "out_baseline_oa1_50", "baseline_17500.npz")
CARRIED = "akita_black_bowl_1"
SWEEP = [("SAMPLE_SURFACE", 0.015), ("SAMPLE_SURFACE", 0.020), ("SAMPLE_SURFACE", 0.022),
         ("SAMPLE_SURFACE", 0.025), ("SAMPLE_SURFACE", 0.030),
         ("VOXEL_SURFACE", 0.020), ("VOXEL_SURFACE", 0.025)]


def sample_hull_surface(vertices, faces, n, rng):
    """n points uniform by area over the triangles of a convex hull."""
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int)
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    p = area / area.sum()
    idx = rng.choice(len(F), size=n, p=p)
    u = rng.random(n)
    v = rng.random(n)
    over = u + v > 1.0
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])


def attached_spheres(router, q, kc):
    """The attached_object link's spheres at configuration q, in the robot base frame."""
    qt = torch.as_tensor(np.asarray(q, np.float32).reshape(1, -1), device=router.tdt.device)
    sph = router.mg.kinematics.get_state(qt).link_spheres_tensor.view(-1, 4)
    s = sph.detach().cpu().numpy()
    s2l = kc.link_sphere_idx_map.detach().cpu().numpy()
    want = kc.link_name_to_idx_map["attached_object"]
    keep = (s2l == want) & (s[:, 3] > 0.0)
    return s[keep]


def main():
    base = {k: np.asarray(v) for k, v in
            dict(np.load(BASELINE, allow_pickle=True)).items() if k.endswith("__eef")}
    env, dom, mon, sg, task, init, obs = D.make_env(TAG, INIT, res=84)
    D.set_render(dom, False)
    for _ in range(D.CONF["SETTLE_STEPS"]):
        obs, *_ = env.step([0.0] * 6 + [-1.0])
    obstacle = D.SPEC["names"][TAG]["obstacle"]
    info = D.eval_rule_shift(sg, obstacle, base[f"{TAG}__i{INIT:02d}__eef"], 0.05)
    shift = np.asarray(info["shift"], float)
    _, _, anch = geom_shift.obstacle_anchors(dom)
    st = geom_shift.apply_shift(dom, anch, init[INIT], shift)
    dom, mon, sg, obs, _ = D.setup_scene(env, st, INIT, shift=shift, render=False)

    home = np.asarray(obs["robot0_eef_pos"], float)
    hq = np.asarray(obs["robot0_eef_quat"], float)
    wp = D.nominal_plan(sg, dom, TAG, home, hq)
    lift = next(w for w in wp if str(w[2]).startswith("lift:"))
    base_p, base_R = W.robot_base_offset(dom)
    meshes = W.scene_meshes(dom, mon, sg, TAG, carried=CARRIED)

    router = R.CuroboRouter.get()
    kc = router.robot_cfg.kinematics.kinematics_config
    eef_now = np.asarray(
        dom.sim.data.site_xpos[dom.sim.model.site_name2id("gripper0_grip_site")], float)
    router.calibrate(dom, eef_now, base_p, base_R)
    router.update_for_scene(meshes)
    quat = R.xyzw_to_wxyz(np.asarray(lift[4], float))
    b0 = router.site_to_hand(W.to_base(np.asarray(lift[0], float).reshape(1, 3),
                                       base_p, base_R)[0], quat)
    q0, ok0 = router.ik(b0, quat)
    payload = W.carried_mesh_abs(dom, mon, CARRIED)
    pv, pf = payload["vertices"], payload["faces"]
    rng = np.random.default_rng(0)
    pts = sample_hull_surface(pv, pf, N_SAMPLES, rng)

    print("=" * 96)
    print(f"PAYLOAD COVERAGE -- {CARRIED}, {N_SAMPLES} points on the true hull surface")
    print("=" * 96)
    print(f"  true hull: {len(pv)} vertices, {len(pf)} faces, extent "
          f"{np.round(pv.max(axis=0) - pv.min(axis=0), 4)} m")
    print(f"  start IK at the lift: {ok0}")
    print()
    print(f"  {'fit':>30} {'radius m':>9} {'spheres':>8} {'max gap cm':>11} "
          f"{'p99 gap':>9} {'median':>8} {'covered %':>10}")
    rows = []
    for fit_name, radius in SWEEP:
        router.detach_payload()
        try:
            att = router.attach_payload(q0, payload,
                                        fit=getattr(SphereFitType, fit_name), radius=radius)
        except Exception as e:
            print(f"  {fit_name:>30} {radius:>9.3f}  attach raised {type(e).__name__}")
            continue
        s = attached_spheres(router, q0, kc)
        if not len(s):
            print(f"  {fit_name:>30} {radius:>9.3f}  no spheres attached (att={att})")
            continue
        d = np.linalg.norm(pts[:, None, :] - s[None, :, :3], axis=2) - s[None, :, 3]
        gap = d.min(axis=1)
        covered = float((gap <= 0).mean()) * 100.0
        print(f"  {fit_name:>30} {radius:>9.3f} {len(s):>8} {100 * gap.max():>11.3f} "
              f"{100 * np.percentile(gap, 99):>9.3f} {100 * np.median(gap):>8.3f} "
              f"{covered:>10.1f}")
        rows.append(dict(fit=fit_name, radius_m=radius, n_spheres=int(len(s)),
                         max_gap_cm=round(float(100 * gap.max()), 4),
                         p99_gap_cm=round(float(100 * np.percentile(gap, 99)), 4),
                         median_gap_cm=round(float(100 * np.median(gap)), 4),
                         covered_pct=round(covered, 2),
                         sphere_r_min_cm=round(float(100 * s[:, 3].min()), 3),
                         sphere_r_max_cm=round(float(100 * s[:, 3].max()), 3)))
    router.detach_payload()
    env.close()

    good = [r for r in rows if r["max_gap_cm"] <= 0.0]
    print()
    print("  gap > 0 means the hull sticks out of the sphere set: UNDER-COVERED.")
    if good:
        best = min(good, key=lambda r: (r["n_spheres"], r["radius_m"]))
        names = ", ".join("{}@{}".format(r["fit"], r["radius_m"]) for r in good)
        print(f"  fits with no under-coverage (max gap <= 0): {names}")
        print(f"  cheapest adequate fit: {best['fit']} radius {best['radius_m']} m, "
              f"{best['n_spheres']} spheres, max gap {best['max_gap_cm']:.3f} cm")
    else:
        worst = min(rows, key=lambda r: r["max_gap_cm"])
        print(f"  NO fit in the sweep covers the hull. Closest: {worst['fit']} "
              f"radius {worst['radius_m']} m, max gap {worst['max_gap_cm']:.3f} cm")
    json.dump(rows, open(os.path.join(OUT, "diag_payload_fit.json"), "w"), indent=1)
    print("DIAG_PAYLOAD_FIT_DONE")


if __name__ == "__main__":
    main()
