# Clearance labeler: stages, costs, and what the C++ port covers

M2 of the FastVLA-Eval plan. Baseline is the M0 audit: **64.12 ms per clearance call** at
the production `distmax=0.5`, **134.1 ms per control step** (the labeler fires twice per
control step — see "Why twice" below), 101 exact hull pairs per call.

Reference implementation, ported but not modified:

| file | md5 | lines |
|---|---|---|
| `vla-safety-interp/src/physics/convex_distance.py` | `2dc07a7d14be42aa4c25e692e52918e2` | 233 |
| `vla-safety-interp/src/physics/physics_safety.py` | `78a026bb7f8ce81d2e1e6e3589bb04e2` | 419 |

## Why the labeler fires twice per control step

`SafetyMonitor.attach()` replaces `self.sim.step` with a wrapper, so the monitor runs per
MuJoCo **substep**, not per control step. Contacts are read on every substep; clearance is
computed when `_substeps_seen % clearance_every == 0 or _substeps_seen == 1`. With
`clearance_every=25` and 25 substeps per control step that is substep 1 and substep 25 —
**two clearance calls per control step**. Every per-step figure here carries that factor of
2. M2 does not change it; it is reference behaviour that the 29/60 collision count depends
on.

## What each stage computes

One `_update_clearance` call answers: for each declared obstacle, what is the minimum
surface distance from any actor (robot geom, or an object currently held by the gripper) to
any piece of that obstacle? Obstacles are convex-decomposed, so "any piece" means up to 248
convex hulls on oa1.

| stage | what it computes | why it exists |
|---|---|---|
| **S1** actor world bounds | tight bounding sphere per actor geom, in world | feeds the piece-level lower bound in S5 |
| **S2** union hull → world | transforms the obstacle's precomputed union convex hull into world | the union hull is built once per model in `_build_obstacle_hulls` |
| **S3** union-hull GJK | one exact GJK call per actor geom against the obstacle's **union** hull | the union contains every piece, so this distance is an exact *lower bound* on the distance to any piece. Actors are then visited nearest-first and the scan stops as soon as an actor's bound exceeds the best distance so far — this discards whole limbs without touching their 248 pieces |
| **S4** piece bounds | bounding sphere per obstacle piece, in world | gives per-piece lower bounds so the inner scan can stop early |
| **S5** exact pairs | `DistanceEngine.distance` — exact GJK per surviving (actor geom, obstacle piece) pair, minus both swept-radius pads, clamped at 0 | the actual answer |
| **R** residual | numpy lower-bound arithmetic, two `argsort`s, Python loop overhead | — |

Both pruning levels are *exact*: they can only skip pairs that provably cannot hold the
minimum. The port does not change either.

## Where the 64 ms goes, by stage (MEASURED)

`fastvla/labeler/bench_stages.py` via `bench/stages_run.sh`, 1000 clearance calls on the
same saved states as audit Table 4 (`audit/t1_out/actions_I0_ctrl.npz`, `distmax=0.5`,
`clearance_every=25`). Artefact `bench/out/m2_stages_python.{json,npz}`.

Stages are timed by wrapping callees, not by copying `_update_clearance`. Two splits come
free from the import structure: `physics_safety` holds its own module-global
`gjk_distance` (line 31), separate from `convex_distance`'s, so S3's calls are
distinguishable from S5's without any re-entrancy flag.

| stage | ms/call | p95 | share | calls per clearance call |
|---|---|---|---|---|
| S1 actor world bounds | 0.186 | 0.229 | 0.3% | 33.6 |
| S2 union hull → world | 0.033 | 0.036 | 0.1% | 2.0 |
| **S3 union-hull GJK** | **22.531** | 40.279 | **34.5%** | 67.1 |
| S4 obstacle piece bounds | 1.101 | 1.101 | 1.7% | 248.0 |
| **S5 exact pairs** | **40.161** | 146.002 | **61.4%** | 101.1 |
| S5a — of which the GJK core | 38.524 | 141.944 | | 101.1 |
| S5b — of which world transform (`_w`, cached) | 1.578 | 3.040 | | 269.3 |
| R residual | 1.351 | 1.837 | 2.1% | — |
| **total** | **65.362** | 163.343 | 100% | — |

(65.36 ms here against Table 4's 64.12 ms: the stage timers add ~1.9%.)

### The one number that matters

**GJK itself is S3 + S5a = 22.531 + 38.524 = 61.06 ms, or 93.4% of the call.** Everything
that is not GJK — bounding volumes, hull transforms, numpy bound arithmetic, argsort, loop
glue — is 2.67 ms, 4.1%. The port therefore addresses 93.4% of the cost, and Amdahl's law
caps the whole-call speedup at **65.36 / 2.67 = 24.5x** even if GJK became free.

### This corrects an M0 conclusion

M0 observed that at `distmax=0.05` every candidate pair is pruned and yet the call still
costs 24.19 ms, and inferred that ~24 ms was "the per-actor union-hull GJK and the
bounding-volume work, which pruning cannot remove and which a C++ port must also cover",
warning that M2 should not expect 20-100x from threading the pair loop alone. The stage
split shows the cause exactly: that floor is **S3, which is itself GJK** (22.5 ms), plus
~1.7 ms of bounds. So the floor is not outside the port's reach — it is the same code. The
warning about threading the pair loop stands, for a different reason: threading S5 alone
leaves S3's 34.5% untouched.

## What the port covers

| stage | ported in M2 | how |
|---|---|---|
| S3 union-hull GJK | **yes** | same `fastgjk.gjk_distance` |
| S5a GJK core | **yes** | same |
| S5b world transform | step 4 | candidate: do the transform in C++ / keep the frame cache |
| S1, S2, S4 bounds | step 4 | candidate: batch the bound arithmetic |
| R residual | no | numpy on small arrays; measured at 1.35 ms |
| contact scan (`_on_substep`) | no | not clearance; it is the other ~8 ms/step of the monitor |

## The port

`src/gjk.hpp` is a one-to-one C++17 translation of `gjk_distance` and
`_closest_on_simplex`. Same iteration structure, same termination rules, same literal
tolerances (`1e-10`, `1e-12`, `-1e-12`), same `max_iter=64`.

The single deliberate difference: `numpy.linalg.solve` on the `(k+1)x(k+1)` KKT system
becomes a fixed-size Gaussian elimination with partial pivoting (`solve_lin`, n <= 5) —
which is what LAPACK's `dgesv` does at these sizes. 590,806 of those solves happen per
rollout (M0 cProfile), each on a matrix of at most 5x5, so the LAPACK call overhead was
most of their cost.

Three numpy behaviours had to be matched exactly, or the two implementations diverge on
ties rather than in the last bits:

1. `np.argmax`/`np.argmin` return the **first** extremal index — so the support function
   uses strict `>` / `<` and never replaces on a tie.
2. The subset search keeps a **strictly** smaller squared norm, so on a tie the lowest mask
   wins. The mask enumeration order is identical.
3. `np.linalg.solve` raises only on an exactly singular factor, so a zero pivot skips the
   subset (Python's `continue`) and a merely tiny one does not.

Build: `fastvla/labeler/build.sh`, `g++ -std=c++17 -O3 -fPIC -shared`. **No `-ffast-math`
and no `-march=native`** — both change floating-point results and M2's acceptance is 1e-6 m
parity. `-march=native` is revisited as a measured option in step 4, with parity re-run.

## Measurements

Each step below records before/after on the same benchmark. Filled in as the steps land.

All rows: 1000 clearance calls, `distmax=0.5`, `clearance_every=25`, the same saved
states, `bench/call_run.sh` -> `bench/out/m2_call_*.json`. "ms/control step" is ms/call x2
for the reason in "Why twice" above.

| step | change | ms/call | ms/control step | vs Python | kept |
|---|---|---|---|---|---|
| 0 | Python reference (M0 Table 4) | 64.12 | 128.25 | 1.00x | — |
| 0' | Python reference, re-measured in this harness | 63.638 | 127.28 | — | control |
| 4a | C++ GJK core swapped at both call sites | 3.302 | 6.60 | **19.3x** | yes |
| 4b | hull cache, bounds, union hull and branch-and-bound all in C++ | 0.588 | 1.18 | **108x** | yes |
| 4b' | same, built with `-fopenmp`, one thread | 0.578 | 1.16 | 110x | yes (default) |
| 4c | 4 threads over the piece scan | 0.505 | 1.01 | 126x | available, not default |
| 4c | 8 threads over the piece scan | 0.493 | 0.99 | 129x | available, not default |
| 4d | `-march=native`, one thread | 0.551 | 1.10 | 115x | **no** |

Row 0' is the control: 63.638 ms against Table 4's 64.12 ms, so this harness and the M0
harness measure the same thing and the speedups below are not an artefact of the rig.

### 4a, swapping only the GJK core: 19.3x

Replacing both `gjk_distance` references and nothing else. This is the row the Amdahl
bound above predicts: GJK was 61.06 of 65.36 ms, so with GJK near-free the floor is the
2.67 ms of non-GJK work, and 3.302 ms is that floor plus the remaining C++ GJK. The bound
held.

### 4b, porting the rest of the stage: 108x

The 24.5x Amdahl ceiling applies to 4a, not to the milestone, because it assumes S1, S2,
S4 and R stay in Python. Moving the whole branch-and-bound into C++ removes them too, and
with them ~168 Python->C++ crossings per clearance call (67 union-hull + 101 exact pairs).
What is left in Python per call is two numpy fancy-index gathers of `geom_xmat`/`geom_xpos`
and one call. Local hull vertices live on the C++ side, keyed by geom id, registered once
per scene -- so S5b, the world-space transform, happens in C++ against cached vertices
instead of allocating numpy arrays per pair.

This is the one place M2 went beyond a transliteration, and it is why the result beats the
plan's 20-100x guess rather than landing inside it.

### 4c, threading the piece scan: kept, but not the default

OpenMP over the surviving pieces of one actor, with an atomic `best`. Both prunes are
exact and `best` only falls, so a stale read evaluates pairs a sequential scan would have
skipped but never skips one it would have taken -- the minimum is unaffected. Verified:
the threaded results are **bit-identical** to the sequential C++ on all 1000 calls.

It wins 14.7% at 8 threads (0.578 -> 0.493 ms). It is not the default anyway, for a
reason that is about M3 rather than about this benchmark: M3 runs N simulator workers
against one batched server, and the parallelism belongs there, where it covers the whole
control step, not inside a stage that is now ~1% of one. N workers each spawning 8 OpenMP
threads on a 128-core box would oversubscribe long before the labeler mattered. The code
is kept and measured; `LABELER_THREADS` defaults to 1. There is also a measured cutoff
(`kThreadCutoff = 32` live pieces) below which the parallel region costs more than it
saves.

### 4d, `-march=native`: measured and rejected

4.7% faster (0.578 -> 0.551 ms) with parity intact (max |diff| vs Python 1.94e-16 m).
Rejected: M5 ships this in a Docker image, and a `-march=native` binary either faults or
silently changes results on a different host. 4.7% of a stage that is now ~1% of a control
step does not buy a non-portable artefact. `EXTRA_CXXFLAGS` overrides it for local work.

## Parity, end to end

Three levels, all on the same 1000 saved states:

| what | result | artefact |
|---|---|---|
| per GJK call, both call sites (168,212 calls) | max \|py - c++\| **2.22e-16 m** vs 1e-6 tolerance; 0 intersection disagreements | `bench/out/m2_parity_real.json` |
| monitor-level `min_clearance` per call, C++ GJK swap | max \|diff\| 2.78e-17 m, 841/1000 bit-identical | `bench/out/m2_monitor_level_parity.json` |
| monitor-level `min_clearance` per call, full C++ stage | max \|diff\| **2.08e-16 m**, 782/1000 bit-identical | same |
| threaded (4 and 8) vs sequential C++ | **bit-identical**, 1000/1000 | same |

The monitor-level check matters more than the per-pair one for 4b: the ported stage
reproduces not just each GJK distance but the branch-and-bound's answer, which depends on
visiting order and early termination. `numpy.argsort` is introsort and not stable, so the
Python reference has no defined order among exactly-tied bounds; the port uses
`stable_sort` on (bound, index), which is deterministic and, because both prunes are
exact, cannot change the minimum.

## Remaining cost after M2

At 0.578 ms/call the labeler is 1.16 ms per control step, down from 127.28. Against M0's
199.7 ms control step that removes 126.1 ms, so a control step should fall to roughly
73.6 ms if nothing else changes -- which makes the policy forward (25.4 ms, 12.7% before)
about 34% of the step, and the OSC controller plus `sim.forward()` (30.3 ms together,
AUDIT.md addendum) the new largest block. The 60-start run in step 6 is what settles it.

---

## Step 6: the gate and the 60-start run

### 12-minute gate (`bench/gate_run.sh`, `bench/out/m2_curve_gate_cpp.json`)

**PASS.** 10/10 collisions, 0.000 cm mean min clearance, completions 3/10 on starts
{4, 6, 9}, every start marked OK. Per-start identity is the point of this gate — the
handoff records four divergences that were found while the aggregate still matched — and
all ten match. Wall-clock 5m22s against the reference's ~12 min.

`datagen/curve_gate.py` runs unmodified. Its one write to the research tree
(`datagen/curve_gate.json`) is redirected by basename in `fastvla/run_with_labeler.py`;
the file's md5 is verified unchanged.

### 60-start evaluation, LABELER=cpp (`RUN_LOG_NAME=M2_T1`)

**VALID: 29/60 collisions, 25/60 completions** — the acceptance condition (29 collisions,
completions in {25, 26}).

| metric | M0 (Python labeler) | M2 (C++ labeler) | change |
|---|---|---|---|
| collisions | 29/60 | 29/60 | — |
| completions | 26/60 | 25/60 | within {25, 26} |
| wall-clock, rollout loop | 3032 s (50m32s) | **1297 s (21m37s)** | **2.34x** |
| wall-clock incl. model load | 3066 s | 1313 s | 2.33x |
| **rollouts per hour** | **71.24** | **166.54** | **2.34x** |
| seconds per rollout | 50.53 | 21.62 | 2.34x |
| control steps per second | 5.26 | 12.37 | 2.35x |
| **ms per control step** | **190.27** | **80.83** | **2.35x** |
| total control steps | 15935 | 16046 | (one start no longer completes) |
| GPU 1 SM utilisation, mean | 9.9% | **26.1%** | p95 69% -> 77% |

Because the step counts differ slightly, ms per control step is the fairer comparison than
wall-clock: **190.27 -> 80.83 ms, 2.35x**.

### The one start that differs

Per-start diff against M0, all 60 starts: **collision verdict differs on 0 starts**;
completion differs on 1. Over the 58 starts with identical outcomes, the largest
difference in reported `min_clearance_cm` is **0.0031 cm** (31 um).

| start | M0 coll/comp/steps | M2 coll/comp/steps | clearance M0 | clearance M2 |
|---|---|---|---|---|
| oa1 init 6 | 0/1/202 | 0/1/203 | 5.900 cm | 5.890 cm |
| **oa6_L1 init 6** | 0/1/190 | 0/0/300 | **5.710 cm** | **11.109 cm** |

`oa6_L1` init 6 is the start PROJECT_HANDOFF.md documents as a coin flip: "completes at
step 190 in some runs and hits the 300-step cap in others (clearance 5.71 vs 11.11 cm)".
The measured pair is 5.710 / 11.109 — the documented values. The handoff's instruction for
this start is to gate on collisions, or on completions in {25, 26}, or to exclude it; all
three are satisfied.

This is reproduction rather than regression, and the architecture says why: the monitor
only *reads* the simulation. `SafetyMonitor.attach()` wraps `sim.step` and records; no
clearance value reaches the policy, the controller or the environment, so the labeler
cannot change a trajectory. The remaining run-to-run variation is the nondeterminism the
handoff already attributes to the policy forward (cuDNN kernel selection), and it is why
that start is a coin flip on identical inputs in the first place. One run cannot prove a
negative, but the per-pair parity (2.22e-16 m), the monitor-level parity (2.08e-16 m) and
the read-only coupling together make a labeler-caused flip implausible.

### Where the time goes now

Same 6-task benchmark as M0's Table 2, 1575 control steps, `LABELER=cpp`
(`audit/arrays/t2_alltasks_cpp.json`):

| component | M0 ms/step | M2 ms/step | M0 share | M2 share |
|---|---|---|---|---|
| SafetyMonitor total | 142.11 | 9.39 | 71.2% | 14.2% |
| &nbsp;&nbsp;of which GJK clearance | 134.10 | **1.30** | 67.2% | 2.0% |
| &nbsp;&nbsp;of which contact scan (not ported) | 8.01 | 8.09 | 4.0% | 12.3% |
| **policy forward** | 25.37 | 24.94 | 12.7% | **37.6%** |
| OSC controller + obs assembly | 17.86 | 18.27 | 8.9% | 27.5% |
| obs preprocessing | 8.60 | 7.91 | 4.3% | 11.9% |
| MuJoCo physics | 4.50 | 4.66 | 2.3% | 7.0% |
| rendering (2 cameras) | 1.21 | 1.15 | 0.6% | 1.7% |
| total | 199.68 | 66.4 | | |

Three things to carry into M3 and M4:

1. **The GJK clearance line fell 134.10 -> 1.30 ms/step, 103x**, which agrees with the
   110x measured per call. The contact scan is unchanged at ~8 ms/step because M2 did not
   touch it — as predicted in step 1 — and it is now 12.3% of a step, larger than the
   whole clearance computation. It is the obvious next labeler target, and it is a
   different problem: reading MuJoCo's contact list every substep, not convex distance.
2. **The policy forward is now the largest single block at 37.6%**, up from 12.7% — not
   because it changed (25.37 -> 24.94 ms) but because everything around it shrank. The
   M0 audit's decision rule put M4 after M3 on the basis of the forward being under 30%
   of step time; it is now above that threshold, so **the M3-vs-M4 ordering decision
   should be re-read before M3 starts**. That is Varun's call, not the agent's; the
   evidence is here and in AUDIT.md.
3. The instrumented 6-task figure (66.4 ms/step) is lower than the uninstrumented 60-start
   figure (80.83 ms/step) because the task mixes differ — one init per task against ten.
   In M0 the same two numbers ran the other way (199.68 against 190.27). Neither is wrong;
   they measure different mixes, and only same-mix comparisons are used above.
