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

| step | change | ms/call | ms/control step | vs Python | kept |
|---|---|---|---|---|---|
| 0 | Python reference (M0 Table 4) | 64.12 | 128.25 | 1.00x | — |
