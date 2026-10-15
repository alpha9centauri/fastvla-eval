"""Table 3: GPU idle-gap summary from the Nsight Systems CUDA kernel trace.

Answers the plan's question "is the GPU idle while MuJoCo steps? how long are the gaps?"
by merging the kernel intervals from nsys's cuda_gpu_trace report and measuring the holes
between them. A gap is wall-clock time inside the traced window with no kernel resident
on the device.

Env: CSV (an nsys cuda_gpu_trace csv), OUT
"""
import glob
import json
import os

import numpy as np
import pandas as pd

CSV = os.environ.get("CSV", "")
OUT = os.environ["OUT"]
if not CSV:
    c = sorted(glob.glob("/data_new/vchandr2/fastvla-eval/audit/traces/*cuda_gpu_trace*.csv"))
    if not c:
        raise SystemExit("no cuda_gpu_trace csv found; run t3_nsys.sh first")
    CSV = c[-1]

df = pd.read_csv(CSV)
cols = {c.lower().strip(): c for c in df.columns}


def pick(*names):
    for n in names:
        if n in cols:
            return cols[n]
    raise KeyError(f"{names} not in {list(df.columns)}")


start = pd.to_numeric(df[pick("start (ns)", "start")], errors="coerce")
dur = pd.to_numeric(df[pick("duration (ns)", "duration")], errors="coerce")
ok = start.notna() & dur.notna()
s = start[ok].to_numpy(dtype=np.float64)
d = dur[ok].to_numpy(dtype=np.float64)
order = np.argsort(s)
s, d = s[order], d[order]
e = s + d

# merge overlapping/concurrent kernel intervals, then the holes are the idle gaps
merged = []
cs, ce = s[0], e[0]
for i in range(1, len(s)):
    if s[i] <= ce:
        ce = max(ce, e[i])
    else:
        merged.append((cs, ce))
        cs, ce = s[i], e[i]
merged.append((cs, ce))
M = np.array(merged, dtype=np.float64)

window = M[-1, 1] - M[0, 0]
busy = float((M[:, 1] - M[:, 0]).sum())
gaps = M[1:, 0] - M[:-1, 1]
gaps = gaps[gaps > 0]

res = dict(
    source_csv=CSV, n_kernel_rows=int(len(s)), n_busy_intervals=int(len(M)),
    traced_window_sec=window / 1e9,
    gpu_busy_sec=busy / 1e9,
    gpu_busy_pct=100.0 * busy / window if window > 0 else None,
    gpu_idle_sec=(window - busy) / 1e9,
    gpu_idle_pct=100.0 * (window - busy) / window if window > 0 else None,
    n_gaps=int(len(gaps)),
    gap_ms_mean=float(gaps.mean() / 1e6) if len(gaps) else None,
    gap_ms_p50=float(np.percentile(gaps, 50) / 1e6) if len(gaps) else None,
    gap_ms_p95=float(np.percentile(gaps, 95) / 1e6) if len(gaps) else None,
    gap_ms_max=float(gaps.max() / 1e6) if len(gaps) else None,
    gap_ms_total=float(gaps.sum() / 1e6) if len(gaps) else None,
    gaps_over_10ms=int((gaps > 1e7).sum()),
    gaps_over_50ms=int((gaps > 5e7).sum()),
    kernel_time_sec=float(d.sum() / 1e9),
)

# The 86k "gaps" are mostly micro-holes between consecutive kernels inside one forward
# (p50 ~1 us), which is not what the plan is asking about. The macro gaps are the windows
# where the device has nothing at all to do because the process is stepping MuJoCo and
# running the labeler. Thresholded separately so the two are not averaged together.
for thr_ms, label in ((10.0, "macro10"), (50.0, "macro50")):
    g = gaps[gaps > thr_ms * 1e6]
    res[f"{label}_n"] = int(len(g))
    res[f"{label}_total_sec"] = float(g.sum() / 1e9) if len(g) else 0.0
    res[f"{label}_pct_of_window"] = (100.0 * g.sum() / (M[-1, 1] - M[0, 0])) if len(g) else 0.0
    res[f"{label}_ms_mean"] = float(g.mean() / 1e6) if len(g) else None
    res[f"{label}_ms_p50"] = float(np.percentile(g, 50) / 1e6) if len(g) else None
    res[f"{label}_ms_p95"] = float(np.percentile(g, 95) / 1e6) if len(g) else None
    res[f"{label}_ms_max"] = float(g.max() / 1e6) if len(g) else None
# excluding the single startup gap (model load / MuJoCo XML compile) from macro50
g50 = np.sort(gaps[gaps > 5e7])
if len(g50) > 1:
    g = g50[:-1]
    res["macro50_excl_largest_n"] = int(len(g))
    res["macro50_excl_largest_ms_mean"] = float(g.mean() / 1e6)
    res["macro50_excl_largest_ms_p50"] = float(np.percentile(g, 50) / 1e6)
    res["macro50_excl_largest_ms_max"] = float(g.max() / 1e6)
np.savez_compressed(OUT + ".npz", gap_ns=gaps, busy_intervals=M)
json.dump(res, open(OUT + ".json", "w"), indent=2)
for k, v in res.items():
    print(f"[T3] {k}: {v}")
print("T3 GAPS DONE", flush=True)
