"""Render the PLANNER attempt at oa1__i02__c7, frames kept even when it fails to certify.

`step4b_run.run_full` returns before its own save when the executor fails, so a failed attempt
leaves no frames. Nothing in it is edited: `dgcore.Episode` is wrapped so the instance is captured
and its frames survive the early return. Frames come out of Episode.add already rotated 180
degrees, which is the orientation the training pipeline uses.

Author: Varun Chandra. Env: planner, GPU 1.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLN = os.path.dirname(HERE)
DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (PLN, DISP, os.path.join(DISP, "datagen"), "/data_new/vchandr2",
          "/data_new/vchandr2/fastvla-eval",
          "/data_new/vchandr2/vla-safety-interp/src/physics"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("FLOOR_M", "0.02"); os.environ.setdefault("MODE", "full")
os.environ.setdefault("RES", "256")

import dgcore as D        # noqa: E402
import step4b_run as S    # noqa: E402
import route as R         # noqa: E402
import imageio.v2 as imageio  # noqa: E402

_EPS = []
_Base = D.Episode


class _Keep(_Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _EPS.append(self)


D.Episode = _Keep
S.D.Episode = _Keep

pos = json.load(open(os.path.join(HERE, "kept_positions_single.json")))[0]
R.install("curobo")
D.CONF["CERTIFY_MIN_M"] = S.FLOOR_M
base = {k: np.asarray(v) for k, v in dict(np.load(S.BASELINE, allow_pickle=True)).items()
        if k.endswith("__eef")}
rr = S.run_full(pos["init"], np.asarray([pos["shift"][0], pos["shift"][1], 0.0], float),
                base, D.SPEC["names"]["oa1"]["obstacle"], np.random.RandomState(0))
print("[RENDER] outcome:", rr.get("reason"), "accepted:", rr.get("accepted"),
      "min clearance:", rr.get("dry_min_clearance_cm"), flush=True)
ep = max(_EPS, key=lambda e: len(e.base)) if _EPS else None
if ep is None or not len(ep.base):
    print("[RENDER] no frames captured", flush=True); sys.exit(1)
out = os.path.join(HERE, "clips"); os.makedirs(out, exist_ok=True)
np.savez_compressed(os.path.join(out, "planner_i02c7.npz"),
                    base=np.asarray(ep.base, np.uint8),
                    clearance=np.asarray(ep.clearance), eef=np.asarray(ep.eef),
                    meta=json.dumps(rr, default=str))
imageio.mimsave(os.path.join(out, "planner_i02c7.mp4"), list(ep.base), fps=20)
print(f"[RENDER] wrote {len(ep.base)} frames, min clearance "
      f"{float(np.min(ep.clearance)):.3f} cm", flush=True)
print("RENDER_PLANNER_DONE", flush=True)
