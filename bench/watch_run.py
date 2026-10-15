"""Print a live throughput line for a run in flight. Read-only: it watches the run
directory and the GPU, and changes nothing about either.

Each rollout's record is written the moment that rollout finishes (assumption A1), so
counting the files in `rollouts/` is the progress, and elapsed wall time since this
watcher started gives the rate. Used to record docs/media/harness.gif.

usage: watch_run.py <run_id> <total> [seconds]
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_util(index=1):
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
        return int(out[index])
    except Exception:
        return -1


def main():
    run_id, total = sys.argv[1], int(sys.argv[2])
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    rd = os.path.join(ROOT, "bench", "runs", run_id)
    t0 = time.time()
    done = 0
    while time.time() - t0 < limit:
        el = time.time() - t0
        d = os.path.join(rd, "rollouts")
        done = len([f for f in os.listdir(d) if f.endswith(".json")]) if os.path.isdir(d) else 0
        rate = 3600.0 * done / el if el > 0 and done else 0.0
        bar = "#" * int(20 * done / total) + "." * (20 - int(20 * done / total))
        print(
            f"[WATCH] {int(el // 60):02d}:{int(el % 60):02d}  [{bar}] {done:2d}/{total}"
            f"   {rate:6.1f} rollouts/hour   GPU1 {gpu_util():3d}%",
            flush=True,
        )
        if done >= total:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
