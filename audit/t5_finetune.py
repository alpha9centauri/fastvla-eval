"""Table 5: fine-tune data path throughput, ESTIMATED from the existing run log.

WHY NOT MEASURED FRESH. `datagen/run_ft_liftonly.sh` passes the reference 17500 checkpoint
as --vla_path, and finetune_headinit.py rewrites `config.json` INSIDE that directory
(ft_liftonly.log lines 18-20: "Created backup of original config at: .../config.json.back...",
"Updated config.json at: .../config.json"). This project may not modify anything outside
fastvla-eval, and that directory is the base policy for every parity comparison in the
plan, so a fresh 200-step run is not run. The existing log is also the better sample: it
is a complete 3000-iteration run rather than 200 steps, on the same card.

What the log cannot give: GPU utilisation during training. That stays unmeasured and is
reported as such. It is only needed for the optional M8 (two-GPU DDP), which the plan
says not to start without a second card.

Env: LOG, OUT
"""
import json
import os
import re

import numpy as np

LOG = os.environ.get(
    "LOG", "/data_new/vchandr2/vla-safety-interp/src/displacement/ft_liftonly.log")
OUT = os.environ["OUT"]
BATCH, ACCUM = 1, 8
EFFECTIVE_BATCH = BATCH * ACCUM

txt = open(LOG, errors="replace").read()
sit = np.array([float(m) for m in re.findall(r"([0-9.]+)s/it", txt)], dtype=np.float64)
its = np.array([float(m) for m in re.findall(r"([0-9.]+)it/s", txt)], dtype=np.float64)
if len(its):
    sit = np.concatenate([sit, 1.0 / its[its > 0]])

prog = re.findall(r"(\d+)/(\d+) \[(\d+):(\d{2}):(\d{2})<", txt)
final = None
if prog:
    step, total, h, m, s = prog[-1]
    final = dict(step=int(step), total=int(total),
                 elapsed_sec=int(h) * 3600 + int(m) * 60 + int(s))

res = dict(
    source_log=LOG,
    tag="ESTIMATE (from an existing complete run's log, not re-measured by this audit)",
    recipe="finetune_headinit.py, batch_size 1 x grad_accumulation_steps 8, lr 5e-5, "
           "LoRA r32 a16, 2 images + proprio, libero_safety_oa1_mix TFDS",
    effective_batch=EFFECTIVE_BATCH,
    n_sit_samples=int(len(sit)),
    sec_per_it_mean=float(sit.mean()) if len(sit) else None,
    sec_per_it_p50=float(np.percentile(sit, 50)) if len(sit) else None,
    sec_per_it_p95=float(np.percentile(sit, 95)) if len(sit) else None,
    sec_per_it_min=float(sit.min()) if len(sit) else None,
    sec_per_it_max=float(sit.max()) if len(sit) else None,
    final_progress=final,
    gpu_utilisation_pct="NOT MEASURED -- see module docstring",
)
if len(sit):
    res["samples_per_sec_mean"] = EFFECTIVE_BATCH / float(sit.mean())
    res["samples_per_sec_p50"] = EFFECTIVE_BATCH / float(np.percentile(sit, 50))
if final and final["step"]:
    wall_per_it = final["elapsed_sec"] / final["step"]
    res["sec_per_it_from_wall"] = wall_per_it
    res["samples_per_sec_from_wall"] = EFFECTIVE_BATCH / wall_per_it
    res["steps_200_projected_sec"] = 200 * wall_per_it

np.savez_compressed(OUT + ".npz", sec_per_it=sit)
json.dump(res, open(OUT + ".json", "w"), indent=2)
for k, v in res.items():
    if k not in ("source_log", "recipe"):
        print(f"[T5] {k}: {v}")
print(f"[T5] -> {OUT}.json")
print("T5 ESTIMATE DONE", flush=True)
