# Benchmark

Every row below is read from a JSON artefact under `bench/out/` that a bench script wrote.
`make bench-check` fails if this table and those artefacts disagree, so the page cannot
drift from what was measured. `make bench` regenerates the artefacts; `make bench-table`
re-renders the table from them.

The "before" column is the original single-process harness this project started from. The
"after" column is the harness in this repository.

<!-- BENCH TABLE START -->
Every row measured on **RTX A5000 (GPU 1 of 2)**, driver **530.30.02**, torch **2.2.0+cu121** (constant across rows -- one box, one environment). The commit column is the commit that last touched that row's artefact, i.e. when the number was last measured. Reproduce with `make bench`.

| metric | before | after | how measured | commit |
|---|---|---|---|---|
| rollouts/hour, 60-start eval, one RTX A5000 | 71.2 (M0, single process) | **544.8** | `bench/bench_all.sh` → `bench/out/m4_t1_summary.json` | `28deac3` |
| ms per control step, per worker (8 workers sharing one GPU) | 190.3 (M0, one process) | 197.7 | wall x N_workers / total steps; higher than M0 because 8 workers contend for one GPU | `28deac3` |
| GPU 1 SM utilisation during the eval | 9.9% (M0) | 85.4% | `nvidia-smi dmon -s um`, 1 Hz → `bench/parse_dmon.py` | `28deac3` |
| clearance labeler, ms per call (fires twice per control step) | 63.64 | **0.578** | `bench/call_run.sh` → `bench/out/m2_call_*.json` | `348ba38` |
| clearance labeler, µs per exact hull pair | 634 | 5.7 | same, exact pairs counted per call | `348ba38` |
| MuJoCo contact scan, ms per control step | 8.19 | **2.25** | `audit/t2_run.sh` CONTACTS=0 vs 1, identical instrumentation | `e902380` |
| policy forward, ms per call, batch 1 (bf16, 2 cameras + proprio, 8-step chunk) | — | 167.97 | `bench/fwd_run.sh time baseline` | `28deac3` |
| forward split: 7B language model / vision backbone | — | 130.2 / 33.9 ms | same run, CUDA events per submodule | `28deac3` |
| per request: queue wait / GPU forward / transport+prep, at 8 workers | — | 798.1 / 184.2 / 25.9 ms | worker round trip minus server-reported parts | `28deac3` |
| serving layer: our ZeroMQ server vs Triton 23.07, same model | 544.8 roll/h | 511.2 roll/h | 6% slower: service time +16 ms because `prepare_request` runs inside `execute`; its transport is cheaper, 3 vs 26 ms (`fastvla/server/DESIGN.md`) | `3d8cc12` |
| same eval inside Apptainer (unprivileged, --nv) | 544.8 roll/h (host) | **547.5 roll/h** | `bench/incontainer_run.sh` | `02aa037` |
| cold start, server launch → first request served | — | 15.0 s | same run, image pull excluded | `02aa037` |
| correctness: collisions / completions on the 60 fixed starts | 29 / 26 (M0 reference) | **29 / 25** | `bench/accept.py` (requires 29, and 25 or 26) | `28deac3` |
| correctness: per-start collision-verdict flips vs the reference | — | **0** | same; the run-to-run floor is 0 collision flips (`fastvla/monitor/EVAL.md` section 7) | `28deac3` |
| correctness: largest per-start min-clearance difference vs the reference | — | 0.1530 cm | same | `28deac3` |
| labeler stages (python): union-hull GJK reject / exact pair GJK | — | 22.5 / 40.2 ms | `bench/stages_run.sh` | `b9a22b7` |
<!-- BENCH TABLE END -->

## Reading a few of these

**Rollouts per hour** is the headline. One rollout is one trial: a task, a starting
arrangement, up to 300 control steps. The 60-start evaluation set is the standard test.

**ms per control step, per worker** goes *up* while throughput goes up. Eight workers
share one GPU, so each waits longer for its turn; the set still finishes sooner because
eight are in flight.

**GPU utilisation** is the number that started the project. The original harness left the
card idle about 90% of the time.

**The labeler** computes the distance from the robot to each obstacle, twice per control
step. In the original harness this was the single most expensive thing in the loop — more
expensive than running the 7-billion-parameter model.

**Triton** is NVIDIA's inference server, measured here as an alternative to the ZeroMQ
server in this repository. It is slower end to end and is kept as a comparison, not a
recommendation. `docs/CORRECTNESS.md` explains where the difference goes.
