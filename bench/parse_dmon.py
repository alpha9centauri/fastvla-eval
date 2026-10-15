"""Summarise nvidia-smi dmon samples for a run: GPU 1 utilisation and memory, GPU 0 left alone."""

import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "bench/out/m3_gpu_util.json"
LOGS = "/data_new/vchandr2/fastvla-eval/audit/logs"
res = {}
for f in sorted(os.listdir(LOGS)):
    if not (f.startswith("M3_") and f.endswith("_dmon.txt")):
        continue
    tag = f[len("M3_") : -len("_dmon.txt")]
    sm = {0: [], 1: []}
    mem = {0: [], 1: []}
    with open(os.path.join(LOGS, f), errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                g, s, m = int(p[1]), int(p[2]), int(p[3])
            except ValueError:
                continue
            if g in sm:
                sm[g].append(s)
                mem[g].append(m)
    if not sm[1]:
        continue
    a = np.array(sm[1])
    res[tag] = dict(
        n_samples=int(len(a)),
        gpu1_sm_mean=float(a.mean()),
        gpu1_sm_p50=float(np.percentile(a, 50)),
        gpu1_sm_p95=float(np.percentile(a, 95)),
        gpu1_mem_mean=float(np.array(mem[1]).mean()),
        gpu0_sm_mean=(float(np.array(sm[0]).mean()) if sm[0] else None),
    )
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=2)
print(f"{'run':16s} {'samples':>8s} {'GPU1 sm mean':>13s} {'p95':>6s} {'GPU0 sm':>8s}")
for k, v in res.items():
    print(
        f"{k:16s} {v['n_samples']:>8d} {v['gpu1_sm_mean']:>12.1f}% "
        f"{v['gpu1_sm_p95']:>5.0f}% {v['gpu0_sm_mean']:>7.1f}%"
    )
print(f"-> {OUT}")
