"""Table 4 benchmark, for whichever labeler the LABELER flag selects.

Times `_update_clearance` only -- one wrapper, so the numbers are comparable to audit
Table 4's 64.12 ms/call for the Python reference. Replays the same saved states.

Env: ACTIONS_NPZ, OUT, LABELER, [LABELER_LEVEL, LABELER_THREADS, N_CALLS, DISTMAX]
"""

import json
import os
import sys
import time

DISP = "/data_new/vchandr2/vla-safety-interp/src/displacement"
for p in (
    "/data_new/vchandr2",
    "/data_new/vchandr2/openvla-oft",
    DISP,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import physics_safety as ps  # noqa: E402
import rollout_bootstrap  # noqa: F401, E402

from fastvla.labeler import patch  # noqa: E402
from fastvla.labeler.replay import parse_key, replay_rollout  # noqa: E402

ACTIONS_NPZ = os.environ["ACTIONS_NPZ"]
OUT = os.environ["OUT"]
N_CALLS = int(os.environ.get("N_CALLS", "1000"))
DISTMAX = float(os.environ.get("DISTMAX", "0.5"))
CLEARANCE_EVERY = int(os.environ.get("CLEARANCE_EVERY", "25"))

level = patch.install()
LABELER = os.environ.get("LABELER", "python")
print(
    f"[BENCH] LABELER={LABELER} level={level} threads={os.environ.get('LABELER_THREADS', '1')}",
    flush=True,
)

REC = []
CLR = []
_uc = ps.SafetyMonitor._update_clearance


def _timed(self, dd, actor_of):
    t0 = time.perf_counter()
    try:
        return _uc(self, dd, actor_of)
    finally:
        REC.append(time.perf_counter() - t0)
        if self._acc:
            CLR.append(min(d["min_clearance"] for d in self._acc["per_obstacle"].values()))


ps.SafetyMonitor._update_clearance = _timed

acts_all = dict(np.load(ACTIONS_NPZ, allow_pickle=True))
used = []
for k in sorted(acts_all):
    if len(REC) >= N_CALLS:
        break
    a = acts_all[k]
    if a.ndim != 2 or len(a) == 0:
        continue
    tag, ii = parse_key(k)
    before = len(REC)
    n, _mon = replay_rollout(
        tag,
        ii,
        a,
        distmax=DISTMAX,
        clearance_every=CLEARANCE_EVERY,
        on_step=lambda i: len(REC) < N_CALLS,
    )  # noqa: B023
    used.append(dict(key=k, control_steps=int(n), calls=len(REC) - before))
    print(f"[BENCH] {k}: {len(REC) - before} calls ({len(REC)}/{N_CALLS})", flush=True)

A = np.array(REC[:N_CALLS], dtype=np.float64) * 1000.0
res = dict(
    labeler=LABELER,
    level=level,
    threads=int(os.environ.get("LABELER_THREADS", "1")),
    distmax=DISTMAX,
    clearance_every=CLEARANCE_EVERY,
    n_calls=int(len(A)),
    ms_per_call_mean=float(A.mean()),
    ms_per_call_p50=float(np.percentile(A, 50)),
    ms_per_call_p95=float(np.percentile(A, 95)),
    ms_per_call_max=float(A.max()),
    ms_per_control_step=float(A.mean() * 2.0),
    rollouts_used=used,
)
np.savez_compressed(OUT + ".npz", calls=A, min_clearance=np.array(CLR[:N_CALLS]))
with open(OUT + ".json", "w") as fh:
    json.dump(res, fh, indent=2)
print(
    f"\n[BENCH] {LABELER}/{level}: {len(A)} calls  mean {A.mean():.3f} ms  "
    f"p50 {np.percentile(A, 50):.3f}  p95 {np.percentile(A, 95):.3f}  "
    f"-> {A.mean() * 2:.2f} ms/control step"
)
print(f"[BENCH] -> {OUT}.json")
print("BENCH CALL DONE", flush=True)
