"""Render the BASE POLICY at oa1__i02__c7, the one placement where it genuinely collides.

Same call as Step 2 -- the harness's own run_one with the candidate shift applied verbatim -- with
video on. The harness appends raw observations, while the policy itself sees img[::-1, ::-1]
(libero_utils.get_libero_image), so the frames are rotated here before writing; unrotated clips
were the trap in the Test 3 renders.

Author: Varun Chandra. Env: learn-gpu, GPU 1.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISP = "/data_new/vchandr2/vla_old_research/vla-safety-interp/src/displacement"
OUT = os.environ.setdefault("OUTDIR", "/data_new/vchandr2/data/planner_v2/step2")
for p in ("/data_new/vchandr2/rollout_stubs", "/data_new/vchandr2/scratch/openvla-oft",
          "/data_new/vchandr2/LIBERO-Safety",
          "/data_new/vchandr2/LIBERO-Safety/third_party/robosuite-1.4",
          DISP, os.path.join(DISP, "datagen")):
    sys.path.insert(0, p)
for k, v in (("STEP", "17500"), ("N_INIT", "10"), ("OFFSETS", ""), ("ARMS", ""), ("PROBE", "0"),
             ("WRITE_BASELINE", "0"), ("MUJOCO_GL", "egl"),
             ("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")):
    os.environ.setdefault(k, v)

import rollout_openvla_disp as H  # noqa: E402
import imageio.v2 as imageio      # noqa: E402

pos = json.load(open(os.path.join(HERE, "kept_positions_single.json")))[0]
spec = next(s for s in H.SPECS if s["tag"] == "oa1" and int(s["init_idx"]) == pos["init"])
pl = dict(feasible=True, clamped=False, skipped=False, mode="rule_v2",
          shift=[float(pos["shift"][0]), float(pos["shift"][1]), 0.0],
          offset_m=float(np.hypot(*pos["shift"][:2])),
          achieved_m=float(np.hypot(*pos["shift"][:2])),
          intrusion_m=float(pos["intrude_cm"]) / 100.0)
r, skip = H.run_one(spec, 0.0, "base_render", True, placement=pl)
assert r is not None, skip
res = r["res"]
print(f"[RENDER] collision {res['collision']}  step {res.get('collision_step')}  "
      f"min clearance {res['min_clearance_cm']} cm  complete {res['completion']}  "
      f"frames {len(r['frames'])}", flush=True)
out = os.path.join(HERE, "clips"); os.makedirs(out, exist_ok=True)
frames = [np.asarray(f)[::-1, ::-1] for f in r["frames"]]          # policy's own orientation
imageio.mimsave(os.path.join(out, "base_i02c7.mp4"), frames, fps=20)
np.savez_compressed(os.path.join(out, "base_i02c7.npz"),
                    frames=np.asarray(frames, np.uint8),
                    clr=np.asarray(r["clr"], float), eef=np.asarray(r["eef"], float),
                    meta=json.dumps(res, default=str))
print("RENDER_BASE_DONE", flush=True)
