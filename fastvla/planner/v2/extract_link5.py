"""Replay the 10 oa1 control-arm trajectories and record robot0_link5 beside the end effector.

CPU only, no policy: the recorded actions from out_onp_control/actions_onpcontrol.npz are replayed
through the same scene build the harness uses, and the link5 body position is read at every
control step.

Why link5. DATA_CHAPTER_VERDICT.txt is explicit that on oa1 the obstacle has to sit on the plate
approach, "the segment where the arm descends to release the bowl and where robot0_link5, not the
end effector, is what fouls the bottle". Ranking a candidate obstacle position by its distance to
the EE path alone would therefore mis-site it relative to the link that actually collides. Both
paths are saved; placement rule v2 ranks on the smaller of the two distances and records which
part set the ranking.

Author: Varun Chandra. Env: planner (CPU).
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, "/data_new/vchandr2/rollout_stubs")
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for _p in (DISP, os.path.join(DISP, "datagen"),
           "/data_new/vchandr2/vla-safety-interp/src/physics"):
    sys.path.insert(0, _p)

import dgcore as D  # noqa: E402

TAG = "oa1"
N_INIT = int(os.environ.get("N_INIT", "10"))
SRC = os.path.join(DISP, "out_onp_control")
OUT = os.path.join(SRC, "traj_link5_onpcontrol.npz")
BODY = "robot0_link5"


def main():
    acts = np.load(os.path.join(SRC, "actions_onpcontrol.npz"), allow_pickle=True)
    trajs = np.load(os.path.join(SRC, "traj_onpcontrol.npz"), allow_pickle=True)
    out = {}
    print(f"replaying {N_INIT} {TAG} control-arm trajectories for {BODY}")
    for ii in range(N_INIT):
        key = f"{TAG}__i{ii:02d}__base"
        if key not in acts.files:
            print(f"  {key}: no recorded actions, skipped")
            continue
        rec = np.asarray(acts[key], float)
        ee_ref = np.asarray(trajs[key], float)
        env, dom, mon, sg, task, init, obs = D.make_env(TAG, ii, res=84)
        D.set_render(dom, False)
        sim = dom.sim
        bid = sim.model.body_name2id(BODY)
        site = sim.model.site_name2id("gripper0_grip_site")
        for _ in range(D.CONF["SETTLE_STEPS"]):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        l5, ee = [], []
        for t in range(len(rec)):
            obs, r, dn, info = env.step(np.asarray(rec[t], float).tolist())
            l5.append(np.asarray(sim.data.body_xpos[bid], float).copy())
            ee.append(np.asarray(sim.data.site_xpos[site], float).copy())
            if dn:
                break
        l5 = np.asarray(l5, np.float32)
        ee = np.asarray(ee, np.float32)
        n = min(len(ee), len(ee_ref))
        err = float(np.abs(ee[:n] - ee_ref[:n]).max()) if n else float("nan")
        out[key] = l5
        out[key + "__ee"] = ee
        print(f"  {key}: {len(l5)} steps, link5 z {l5[:,2].min():.3f}..{l5[:,2].max():.3f} m, "
              f"replayed-EE vs recorded-EE max |diff| {1000*err:.3f} mm")
        env.close()
    np.savez_compressed(OUT, **out)
    print(f"\nwrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB, {len(out)//2} trials)")
    print("EXTRACT_LINK5_DONE")


if __name__ == "__main__":
    main()
