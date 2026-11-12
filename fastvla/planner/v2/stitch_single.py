"""Side-by-side clip: base policy (left) against the planner attempt (right), same placement.

Both sources are already rotated 180 degrees into the policy's own orientation before they reach
here. The shorter clip is held on its last frame so the two stay in step. A caption bar names each
half, because in the Test 3 clips which arm was on which side had to be reconstructed afterwards
from the render code.

Author: Varun Chandra. CPU.
"""
import json, os
import numpy as np
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "clips")
b = np.load(os.path.join(C, "base_i02c7.npz"), allow_pickle=True)
p = np.load(os.path.join(C, "planner_i02c7.npz"), allow_pickle=True)
bf = np.asarray(b["frames"], np.uint8)
pf = np.asarray(p["base"], np.uint8)
bm = json.loads(str(b["meta"])); pm = json.loads(str(p["meta"]))
n = max(len(bf), len(pf))


def pad(a):
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)], axis=0)


bf, pf = pad(bf), pad(pf)
h = min(bf.shape[1], pf.shape[1])
w = min(bf.shape[2], pf.shape[2])
bar = 22
out = np.zeros((n, h + bar, 2 * w + 6, 3), np.uint8)
out[:, :bar] = 30
out[:, bar:, :w] = bf[:, :h, :w]
out[:, bar:, w + 6:] = pf[:, :h, :w]
try:
    from PIL import Image, ImageDraw
    lab = Image.new("RGB", (2 * w + 6, bar), (30, 30, 30))
    d = ImageDraw.Draw(lab)
    d.text((6, 6), "BASE POLICY  collides step "
           f"{bm.get('collision_step')}", fill=(255, 120, 120))
    d.text((w + 12, 6), f"cuROBO PLAN  {pm.get('reason','')}"[:46], fill=(255, 220, 120))
    out[:, :bar] = np.asarray(lab)[None]
except Exception as e:
    print("caption skipped:", e)
imageio.mimsave(os.path.join(C, "side_i02c7.mp4"), list(out), fps=20)
print(f"wrote side_i02c7.mp4  {n} frames  base {len(b['frames'])}  planner {len(p['base'])}")
print(f"  base: collision {bm.get('collision')} step {bm.get('collision_step')} "
      f"minclr {bm.get('min_clearance_cm')} cm complete {bm.get('completion')}")
print(f"  plan: planned {pm.get('planned')} via_min {pm.get('via_min_cm')} cm  "
      f"executed minclr {pm.get('dry_min_clearance_cm')} cm  reason {pm.get('reason')}")
print("STITCH_DONE")
