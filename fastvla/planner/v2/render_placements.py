"""Static frames of ten v2 candidate placements, one per init, spread across the bins. CPU only.

Each frame shows the scene with the perfume stack moved to the candidate position and the bowl
set at its goal pose on the plate, so the clear-goal property of rule v2 is visible: the stack
sits on the carry path, not on the destination.

Frames are rotated 180 degrees to match the policy's own view (libero_utils.get_libero_image),
because the harness writes observations un-rotated.

Author: Varun Chandra. Env: planner (CPU).
"""
from __future__ import annotations

import json
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
import geom_shift  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ["OUTDIR"]
TAG = "oa1"
BOWL = "akita_black_bowl_1"
ORDER = ["3-5", "5-7", "7-9", "9-1000"]      # cycled across the ten inits


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = json.load(open(os.path.join(HERE, "candidates_v2.json")))["rows"]
    picks = []
    for ii in range(10):
        want = ORDER[ii % len(ORDER)]
        pool = [r for r in cands if r["init"] == ii and r["bin"] == want] or \
               [r for r in cands if r["init"] == ii]
        picks.append(sorted(pool, key=lambda r: -r["intrude_cm"])[0])

    rows = []
    for r in picks:
        ii = r["init"]
        env, dom, mon, sg, task, init, obs = D.make_env(TAG, ii, res=512)
        D.set_render(dom, True)
        for _ in range(D.CONF["SETTLE_STEPS"]):
            obs, *_ = env.step([0.0] * 6 + [-1.0])
        _, _, anch = geom_shift.obstacle_anchors(dom)
        sim = dom.sim
        # move the whole declared stack to the candidate position
        shift = np.asarray(r["shift"], float)
        for name, (kind, qadr, bid) in anch.items():
            if kind == "free":
                sim.data.qpos[qadr:qadr + 3] = np.asarray(sim.data.qpos[qadr:qadr + 3], float) + shift
            else:
                sim.model.body_pos[bid] = np.asarray(sim.model.body_pos[bid], float) + shift
        # bowl to its goal pose on the plate
        home = np.asarray(obs["robot0_eef_pos"], float)
        hq = np.asarray(obs["robot0_eef_quat"], float)
        wp = D.nominal_plan(sg, dom, TAG, home, hq)
        place = next(w for w in wp if str(w[2]).startswith("place:"))
        tgt = np.asarray(place[0], float)
        kind, adr, _bid = geom_shift.target_anchor(dom, BOWL)
        assert kind == "free", f"{BOWL} is {kind}, expected a free joint"
        base_z = float(sim.data.qpos[adr + 2])
        sim.data.qpos[adr:adr + 3] = [tgt[0], tgt[1], base_z]
        mujoco.mj_forward(sim.model._model, sim.data._data)
        ob = dom._get_observations(force_update=True)
        img = np.asarray(ob["agentview_image"])[::-1, ::-1]          # policy's own orientation
        fn = f"i{ii:02d}_{r['bin'].replace('9-1000','over9')}_{r['intrude_cm']:.1f}cm_{r['ranked_by']}.png"
        imageio.imwrite(os.path.join(OUT, fn), img)
        rows.append(dict(file=fn, init=ii, bin=r["bin"], intrude_cm=round(r["intrude_cm"], 2),
                         ranked_by=r["ranked_by"], d_ee_cm=round(r["d_ee_cm"], 2),
                         d_l5_cm=round(r["d_l5_cm"], 2), x=round(r["x"], 4), y=round(r["y"], 4)))
        print(f"  i{ii:02d}  bin {r['bin']:<8} intrusion {r['intrude_cm']:.2f} cm  "
              f"ranked_by {r['ranked_by']:<5} -> {fn}", flush=True)
        env.close()
    json.dump(rows, open(os.path.join(OUT, "placements.json"), "w"), indent=1)
    print("RENDER_PLACEMENTS_DONE")


if __name__ == "__main__":
    main()
