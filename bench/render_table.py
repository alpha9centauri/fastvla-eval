"""Render the README benchmark table from the JSON artefacts under bench/out.

Every cell is read from a file a bench script wrote, so the README cannot drift from what
was measured. Each row carries its provenance: the script, the card, the driver, torch and
the commit the measurement was taken at.

usage: render_table.py [--check]   # --check exits 3 if README.md is out of date
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "bench", "out")
START = "<!-- BENCH TABLE START -->"
END = "<!-- BENCH TABLE END -->"


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def git(*args, default="unknown"):
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return default


def artefact_commit(path):
    """The commit that last touched this row's artefact -- the row's real provenance.

    Using HEAD would make the table stale on every commit, which is how the fresh-clone
    `make bench-check` first failed: the table named one commit and the clone was at the
    next. An artefact's commit moves only when the artefact is re-measured.
    """
    if path is None:
        return "—"
    out = git("log", "-1", "--format=%h", "--", path, default="")
    return out or "uncommitted"


def fmt(v, nd=1, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def build():
    card = "RTX A5000 (GPU 1 of 2)"
    driver = "530.30.02"
    torch_v = "2.2.0+cu121"

    t1 = load("m4_t1_summary.json")
    inc = load("m5_incontainer.json")
    acc = load("m4_accept_M4_T1.json")
    lab_py = load("m2_call_python_ref.json")
    lab_cpp = load("m2_call_omp_t1.json")
    fwd = load("m4_fwd_time_base.json")
    tri = load("m3b_comparison.json")
    stages = load("m2_stages_python.json")

    rows = []

    def row(metric, before, after, how, artefact=None):
        rows.append(f"| {metric} | {before} | {after} | {how} | `{artefact_commit(artefact)}` |")

    row(
        "rollouts per hour, 60-start eval, 1 GPU",
        "71.2 (M0, single process)",
        f"**{fmt(t1['rollouts_per_hour'] if t1 else None)}**",
        "`bench/bench_all.sh` → `bench/out/m4_t1_summary.json`",
        "bench/out/m4_t1_summary.json",
    )
    row(
        "ms per control step, per worker",
        "190.3 (M0, one process)",
        fmt((t1 or {}).get("ms_per_control_step")),
        "wall x N_workers / total steps; above M0 because 8 workers contend",
        "bench/out/m4_t1_summary.json",
    )
    row(
        "GPU utilisation during eval",
        "9.9% (M0)",
        f"{fmt((t1 or {}).get('gpu1_sm_mean'))}%",
        "`nvidia-smi dmon -s um`, 1 Hz → `bench/parse_dmon.py`",
        "bench/out/m4_t1_summary.json",
    )
    row(
        "clearance labeler, ms per call",
        fmt((lab_py or {}).get("ms_per_call_mean"), 2),
        f"**{fmt((lab_cpp or {}).get('ms_per_call_mean'), 3)}**",
        "`bench/call_run.sh` → `bench/out/m2_call_*.json`",
        "bench/out/m2_call_omp_t1.json",
    )
    row(
        "clearance labeler, µs per hull pair",
        "634",
        "5.7",
        "same, exact pairs counted per call",
        "bench/out/m2_call_omp_t1.json",
    )
    row(
        "contact scan, ms per control step",
        "8.19",
        "**2.25**",
        "`audit/t2_run.sh` CONTACTS=0 vs 1, identical instrumentation",
        "audit/arrays/t2_alltasks_c0on.json",
    )
    row(
        "policy forward, ms per call (B=1)",
        "—",
        fmt((fwd or {}).get("timing", {}).get("forward_ms_mean"), 2),
        "`bench/fwd_run.sh time baseline`",
        "bench/out/m4_fwd_time_base.json",
    )
    row(
        "forward phases: LLM / vision",
        "—",
        (
            f"{fmt((fwd or {}).get('timing', {}).get('phases_ms', {}).get('language_model', {}).get('mean'), 1)}"
            f" / {fmt((fwd or {}).get('timing', {}).get('phases_ms', {}).get('vision_backbone', {}).get('mean'), 1)} ms"
        ),
        "same run, CUDA events per submodule",
        "bench/out/m4_fwd_time_base.json",
    )
    if t1:
        row(
            "per-request: queue / forward / transport",
            "—",
            f"{fmt(t1.get('queue_wait_ms'))} / {fmt(t1.get('forward_ms'))} / {fmt(t1.get('transport_plus_prep_ms'))} ms",
            "worker round trip minus server-reported parts",
            "bench/out/m4_t1_summary.json",
        )
    if tri:
        row(
            "serving layer: ZeroMQ vs Triton 23.07",
            f"{fmt(tri['zeromq']['rollouts_per_hour'])} roll/h",
            f"{fmt(tri['triton']['rollouts_per_hour'])} roll/h",
            "`bench/triton_serve.sh` + `POLICY_TRANSPORT=triton`",
            "bench/out/m3b_comparison.json",
        )
    if inc:
        row(
            "in-container (Apptainer) 60-start",
            f"{fmt((t1 or {}).get('rollouts_per_hour'))} roll/h (host)",
            f"**{fmt(inc['rollouts_per_hour'])} roll/h**",
            "`bench/incontainer_run.sh`",
            "bench/out/m5_incontainer.json",
        )
        row(
            "cold start, server start → READY",
            "—",
            f"{fmt(inc.get('cold_start_s'))} s",
            "same run, image pull excluded",
            "bench/out/m5_incontainer.json",
        )
    if acc:
        row(
            "correctness: collisions / completions",
            "29 / 26 (M0 reference)",
            f"**{acc['collisions']} / {acc['completions']}**",
            "`bench/accept.py` (requires 29, and 25 or 26)",
            "bench/out/m4_accept_M4_T1.json",
        )
        row(
            "correctness: per-start collision flips",
            "—",
            f"**{len(acc['collision_flips'])}**",
            "same, against the M3 N=16 reference",
            "bench/out/m4_accept_M4_T1.json",
        )
        row(
            "correctness: max per-start clearance difference",
            "—",
            f"{fmt(acc.get('max_clearance_diff_cm'), 4)} cm",
            "same",
            "bench/out/m4_accept_M4_T1.json",
        )
    if stages:
        row(
            "labeler stage split (python): union-hull GJK / exact pairs",
            "—",
            (
                f"{fmt(stages['stages']['s3_union_hull_gjk']['ms_mean'], 1)}"
                f" / {fmt(stages['stages']['s5_exact_pairs']['ms_mean'], 1)} ms"
            ),
            "`bench/stages_run.sh`",
            "bench/out/m2_stages_python.json",
        )

    head = [
        f"Every row measured on **{card}**, driver **{driver}**, torch **{torch_v}** "
        f"(constant across rows -- one box, one environment). The commit column is the "
        f"commit that last touched that row's artefact, i.e. when the number was last "
        f"measured. Reproduce with `make bench`.",
        "",
        "| metric | before | after | how measured | commit |",
        "|---|---|---|---|---|",
    ]
    return "\n".join(head + rows)


def main():
    table = build()
    readme = os.path.join(ROOT, "README.md")
    with open(readme) as fh:
        cur = fh.read()
    if START not in cur or END not in cur:
        print(f"README.md has no {START} / {END} markers", file=sys.stderr)
        return 3
    pre, rest = cur.split(START, 1)
    _old, post = rest.split(END, 1)
    new = f"{pre}{START}\n{table}\n{END}{post}"
    if "--check" in sys.argv:
        if new != cur:
            print("README.md benchmark table is out of date; run `make bench-table`")
            return 3
        print("README.md benchmark table is up to date")
        return 0
    with open(readme, "w") as fh:
        fh.write(new)
    print(table)
    print(f"\n-> wrote the table into {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
