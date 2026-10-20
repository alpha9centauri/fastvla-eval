# Step 4b — the two geometry fixes, and what they revealed

Author: Varun Chandra, 2026-10-20. Supplements `RESULTS.md` (step 4), which it corrects in one
important place. GPU 1, one process, one warm cuRobo instance. Artefacts:
`out/step4b_full.json`, `out/step4b_full_report.txt`, `out/sweep_*.log`, `clips/step4b_*`,
log `PLANNER_STEP4B.log`.

## Headline

Both geometry fixes are correct and verified. Planning recovers from 0 to **55 of 129 scenes
(42.6%)**, the planned routes are now genuinely clear of the obstacle (median per-scene minimum via
clearance **2.03 cm**, 50.9% of planned scenes at ≥ 2 cm), and the executor tracks the planner's
own vias to within **5–7 mm**.

**Acceptance is still 0/129 at 2 cm and 0/129 at 1 cm**, and the reason is now precisely located:
all 55 scenes that planned fail in the **final descent to the plate**, which by the brief's design
belongs to dgcore's primitives and not to the planner. Every one of the 55 ends in contact —
executed minimum GJK clearance exactly **0.000 cm**, in 55 of 55.

## Correction to step 4, as requested

`RESULTS.md` attributed step 4's optimism to the payload being **under-covered** by cuRobo's
surface-sphere fit. That was not the main cause. The attachment was **mislocated by 12.44 cm**,
almost entirely in −z: `attach_external_objects_to_robot` derives the gripper-relative transform
from (object pose, joint state) at attach time, the hull is sampled with the bowl still on the
stove, and the call was made at the **lift** configuration — so cuRobo was told the bowl hangs
12.44 cm below the hand, where it stayed for every subsequent configuration, intersecting the
table. Measured: 35 of 125 spheres penetrating by 3.5 cm at a pose where the real bowl has
32.4 cm of clearance to the bottle and 14.4 cm to the table.

So step 4's planner was not reasoning about an under-sized bowl; it was reasoning about a bowl in
the wrong place. Under-coverage was real and also fixed, but it is the second-order error.

## The bug list, for whoever is next

| bug | symptom | why it is easy to hit |
|---|---|---|
| **4-sphere cap on the attached object** | `franka.yml` sets `extra_collision_spheres: {attached_object: 4}`. Four spheres cannot cover an 11×11×5 cm bowl at any radius — every fit attached 3–4 spheres covering 0.2–25% of the hull, hull protruding 4.0–6.7 cm | the number is in the robot config, not the attach call, and nothing warns. cuRobo logs only `only got 3/4 samples!` |
| **attach at the wrong configuration** | 12.44 cm mislocation, planner silently optimistic | the API takes a joint state and it is natural to pass the one you are planning *from*. It must be the one the object is *held at* |
| cuRobo volume fit degenerates | `sphere_fit: occupancy test failed, assuming cuboid volume` → a solid 11×11×5 cm block welded to the gripper, refusing every hand-off | the message is a log line, not an error |
| `MotionGen.solve_ik` after `warmup(enable_graph=True)` | `MotionGenStatus.IK_FAIL`, which reads like a crowded scene | the real error, *"changing goal type, cuda graph reset not available"*, is swallowed |
| `inflate_hull` via half-space intersection | qhull *"a wide merge error … input is nearly degenerate"* killed the sweep at d = 0.005 on the bottle's 824-vertex hull | the tighter construction is the obvious one to write |

## Fix 1 — obstacle inflation

`inflate_hull` displaces each facet's own vertices along that facet's outward normal by `d` and
takes one convex hull of all displaced points. Every original face plane `a·x = b` then has points
at `a·x = b + d` inside the result, so each face moves out by **at least** `d`: the result contains
the true rounded offset, which is the safe direction — the planner ends up stricter than the
certifier, never looser. Verified exact (`d` to 4 decimals) on a 10 cm cube at 0.003/0.005/0.008/
0.020 m, and clean on the near-degenerate flat hull that killed the first version.

Support geometry (table, plate, bystanders) has its own `PLANNER_INFLATE_SUPPORT_M`, default **0**.
This is a deliberate departure from the brief's "bottle, box, plate, table, bystanders": dgcore
certifies clearance to the **declared obstacles** only, and demanding 2 cm of air around the plate
would forbid the task itself.

## Fix 2 — the payload

Budget raised to 64 spheres in the loaded config dict (the installed `franka.yml` is untouched),
attached at the **grasp** configuration, fit chosen by measuring coverage over 1,000 points on the
bowl's true hull surface. `gap = min_i(|p − c_i| − r_i)`; `gap > 0` means the hull sticks out.

| fit | radius | spheres | max gap | covered |
|---|---|---|---|---|
| SAMPLE_SURFACE | 0.020 | 64 | +0.585 cm | 98.1% |
| **SAMPLE_SURFACE** | **0.022** | **64** | **−0.034 cm** | **100%** |
| VOXEL_SURFACE | 0.020 | 64 | −0.200 cm | 100% |
| SAMPLE_SURFACE | 0.020 | 128 | −0.406 cm | 100% |

Chose 64 @ 22 mm — the tightest fit with no under-coverage, so the payload is not over-approximated
for nothing. Placement after the grasp-configuration fix: offset 1.20 cm, bounding boxes agreeing
to ~1 mm (the residual is because the spheres are surface-centred, so the mean of their centres is
not the hull centroid).

## The inflation sweep, n = 24

First 24 of the 129, same seeds, plan-only, corrected payload.

| inflation | planned | all vias min/median/max | per-scene min via | ≥2 cm | ≥1 cm |
|---|---|---|---|---|---|
| 0.000 m | 8/24 (33.3%) | 0.79 / 7.83 / 37.02 | 0.79–3.25, med 2.51 | 5/8 = 62.5% | 7/8 = 87.5% |
| **0.003 m** | **7/24 (29.2%)** | 1.14 / 7.56 / 35.22 | 1.14–3.17, med 2.08 | 4/7 = 57.1% | **7/7 = 100%** |
| 0.005 m | 6/24 (25.0%) | 1.13 / 7.68 / 35.22 | 1.13–3.17, med 1.97 | 3/6 = 50.0% | 6/6 = 100% |
| 0.008 m | 4/24 (16.7%) | 1.71 / 7.33 / 34.15 | 1.71–3.17, med 2.15 | 3/4 = 75.0% | 4/4 = 100% |

**0.003 m chosen**: the largest setting inside the 20% band (7/8 = 87.5% retained; 0.005 falls to
75%). On the stated tie-break it is not strictly best — the ≥ 2 cm fraction dips 62.5% → 57.1% —
but that is one scene out of seven, not distinguishable at this sample size, while 0.003 lifts the
worst-case via from 0.79 to 1.14 cm and takes ≥ 1 cm to 100%. 0.008's 75% is an artefact of having
already discarded half the scenes.

Note against the earlier n=8 probe, which suggested 4/8 = 50% planning at inflation 0: at n=24 it
is 8/24 = 33%. The probe was optimistic and the larger sample is the number to use.

## The 129-scene run at 0.003 m

129 scenes, 96.4 min, real certifier (dry pass, GJK gate, rendered-replay gate).

```
returned a path                        55/129 = 42.6%
planned-via clearance, ALL vias (1136) min 0.00  median 7.98  max 36.58 cm
per-scene min via clearance            min 0.00  median 2.03  max  5.08 cm
planned scenes with min via >= 2 cm    28/55 = 50.9%
planned scenes with min via >= 1 cm    42/55 = 76.4%

ACCEPTED at 1 cm                       0/129
ACCEPTED at 2 cm                       0/129

outcome breakdown
  planner_failure (no hand-off / no path)   74
  executor                                  55
```

**Fix 1's goal is met on half the planned scenes**: 28 of 55 planned routes are ≥ 2 cm clear to the
certifier's own geometry, against 0 of 129 in step 4 where every route measured 0.000–0.072 cm. The
plans are now honest. Acceptance is nonetheless 0, for a reason that is not about the plans.

## Where the 55 die, measured

```
unreached:pre_place    54/55          unreached:place     1/55
executed min GJK clearance            0.000 cm in 55 of 55
worst per-waypoint arrival error      83-200 mm, median 126 mm   (tolerance 12 mm)
steps used before failure             158-213   (per-waypoint budget 60, episode cap 600)
```

Closest approach to each waypoint, median over the 55 scenes:

| waypoint | median closest approach |
|---|---|
| `grasp` | 4.7 mm |
| `lift` | 5.2 mm |
| `detour_curobo_v0` | 5.2 mm |
| `detour_curobo_v26/v27` | 5.9 / 6.9 mm |
| **`pre_place`** | **44.5 mm** |
| **`place`** | **126.4 mm** |

The executor reaches the grasp, the lift and **every one of the planner's transport vias to within
5–7 mm**. It then gets to 44.5 mm of `pre_place` and stops, in contact. So the transport the
planner is responsible for executes correctly, and the leg that fails is the final descent to the
plate — the leg the brief assigns to dgcore's primitives ("grasp pose and place pose come from the
existing primitives") and which the planner is therefore not allowed to plan.

A step-budget explanation was tested and **refused**: `WP_MAX_STEPS = 60` at `STEP_CAP_M = 0.02`
allows 1.2 m of commanded motion per waypoint, against a hand-off-to-place distance of at most
29.4 cm, and the episodes stop at 158–213 steps against a 600-step cap. The arm is not running out
of budget; it is blocked.

This is the same binding segment every previous attempt found: the 2b smoke's 193 failures bound at
`pre_place` 180 times, and `DATA_CHAPTER_VERDICT.txt` says it plainly — the bottle sits in the
carried bowl's descent to the plate. With the geometry now represented correctly, the motion
planner confirms it rather than contradicting it: **the transport is solvable and the terminal
approach is not.**

## What would follow

The planner is excluded from the one leg that fails. The obvious next move is to let it plan into
the place pose as well, with the obstacle and payload as now represented, instead of handing off
above the plate and leaving a straight-line primitive to finish — which on these scenes drives the
bowl into the bottle. That is a change to the brief's division of labour, not a tuning parameter,
and it is a step-5-or-later decision. Nothing here was tuned toward a result.

## Clips

`clips/step4b_*.mp4`, three scenes, the best planned-via clearance among the 55, rendered through
dgcore's own executor and monitor. Nothing accepted, so these are closest-to-accepted.

| clip | init | shift | planned-via min | executed min | outcome |
|---|---|---|---|---|---|
| `step4b_s08a03_i28_shift14cm_executor.mp4` | 28 | 14.4 cm | 21.52 cm | 0.00 cm | unreached `pre_place` |
| `step4b_s06a03_i28_shift16cm_executor.mp4` | 28 | 16.0 cm | 25.35 cm | 0.00 cm | unreached `pre_place` |
| `step4b_s02a09_i10_shift12cm_executor.mp4` | 10 | 12.2 cm | 18.98 cm | 0.00 cm | unreached `pre_place` |

One caveat on these three: the planned-via minima above (19–25 cm) are **better than the values
recorded for the same scenes in the run** (3.6–5.0 cm), because the render re-plans and cuRobo's
trajectory optimisation is not deterministic across processes. The failure reproduces exactly —
unreached `pre_place`, 0.00 cm executed clearance — but the specific path does not, so these clips
illustrate the failure rather than replaying the measured trajectory.
