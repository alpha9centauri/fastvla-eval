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
| `step4b_s08a03_i28_shift14cm_executor.mp4` *(deleted)* | 28 | 14.4 cm | 21.52 cm | 0.00 cm | unreached `pre_place` |
| `step4b_s06a03_i28_shift16cm_executor.mp4` | 28 | 16.0 cm | 25.35 cm | 0.00 cm | unreached `pre_place` |
| `step4b_s02a09_i10_shift12cm_executor.mp4` | 10 | 12.2 cm | 18.98 cm | 0.00 cm | unreached `pre_place` |

The `s08a03` clip was **deleted on 20 Oct**: its frames were an uninitialised EGL camera buffer
(static noise), not the scene; the row's numbers come from the monitor and are unaffected.

One caveat on these three: the planned-via minima above (19–25 cm) are **better than the values
recorded for the same scenes in the run** (3.6–5.0 cm), because the render re-plans and cuRobo's
trajectory optimisation is not deterministic across processes. The failure reproduces exactly —
unreached `pre_place`, 0.00 cm executed clearance — but the specific path does not, so these clips
illustrate the failure rather than replaying the measured trajectory.

---

# 7a final — the planner plans through to the place pose

Author: Varun Chandra, 2026-10-20. The division of labour changed once, as agreed: the planner owns
grasp → place with the payload attached and the receptacle treated as a support body rather than an
obstacle, and dgcore does only the release. Certification at **1 cm**. Payload spheres 12 mm on
surface sampling, obstacle inflation 0.003 m. Artefacts: `out/step4b_full_4c.json`,
`out/place_pose_gap.json`, log `PLANNER_STEP4C.log`.

## Result

```
scenes                129        wall 46.5 min
returned a path        0/129 = 0.0%
ACCEPTED at 1 cm       0/129
ACCEPTED at 2 cm       0/129
failures               planner_failure:curobo:ik_place   129/129
```

Every scene fails at the same point: collision-aware IK cannot place the arm at the place pose with
the bowl in the gripper. No path is produced, so there are no planned-via or executed clearance
distributions, no executor arrival errors, and no accepted demos to render.

## Why — the destination is occupied

Measured directly before the run, on 20 scenes, as the true GJK clearance of the carried bowl at
dgcore's own place pose against each declared obstacle:

```
bowl -> bottle_of_perfume_1      9.79 - 9.84 cm      (ample)
bowl -> white_place_box_1        0.00 cm             in 18 of 20 scenes
min over both obstacles          min 0.00   median 0.00   max 22.38 cm
scenes with >= 0.5 cm            2/20
scenes with >= 1.0 cm            2/20
scenes with >= 2.0 cm            2/20
```

**At the place pose the carried bowl is already touching `white_place_box_1`.** oa1's obstacle is a
stack — the perfume bottle standing on a white box — and the BDDL forbids contact with *both*
parts, so the box is a declared obstacle that certification counts. The eval rule drives that stack
toward the policy's own baseline trajectory, which passes over the plate; the box therefore ends up
adjacent to the plate, and a bowl set down on the plate touches it.

This is not a planning failure, an executor failure, or a margin failure. The **destination has
zero clearance to a declared obstacle in about 90% of these scenes**, so no certification floor
above 0 cm can ever be met there, by any planner, with any margin, however good the geometry. It
also explains every earlier result in 7a: step 4b's 55 executor failures all stalled at
`pre_place`/`place` in contact, the 2b smoke bound at `pre_place` 180 of 193 times, and
`DATA_CHAPTER_VERDICT.txt` said from the start that the bottle sits in the carried bowl's descent to
the plate.

## The closing paragraph, with the correction it needs

The paragraph this run was meant to produce — *"no collision-free route exists at 1 cm margin for a
whole-arm planner with correct geometry at the test's placements"* — is **true but misattributed**,
and the stronger, correct statement is:

> At the displacement test's eval-rule placements on oa1, no certifiable demonstration exists at any
> positive margin, because the task's own place pose puts the carried bowl in contact with a
> declared obstacle. Measured over 20 scenes: bowl-to-box clearance at the place pose is exactly
> 0.00 cm in 18, and 21.1-22.4 cm in the remaining 2; bowl-to-bottle clearance is 9.8 cm throughout.
> A whole-arm planner with verified geometry — obstacle hulls from the certifier's own engine
> inflated 3 mm, the payload covered to within -0.22 cm over 1,000 hull surface points, attached at
> the grasp configuration — returned 0 paths in 129 scenes, failing at the place pose itself in all
> 129. The limit is the scene, not the planner: the eval rule moves the obstacle stack onto the
> destination.

## One thing this run does NOT establish, stated plainly

Two of the 20 measured scenes (both init 34, shifts 19.31 and 18.07 cm) have a place pose that is
**21.1-22.4 cm clear of both obstacles** — genuinely feasible. Both were in the 129 and both were
recorded as `ik_place` failures. Checking the first of them afterwards through the router's own code
path, with the run's exact world (obstacles + table + 5 bystanders, receptacle excluded) and the
payload attached:

```
world OFF                                IK = True
obstacles only                           IK = True
table only                               IK = True
obstacles + table                        IK = True
obstacles + table + bystanders (the run) IK = True
full world including the plate           IK = False
ROUTER ik_collision_free (the run's call) IK = True     payload attached, 384 spheres
```

So the place pose on that scene **is** reachable and collision-free by the very call the run makes,
and its failure in the run is unexplained. The most likely cause is stale attached-payload state in
the long-lived router across scenes — `attach_payload`/`detach_payload` are called once per scene on
a single persistent `MotionGen`, and a payload left attached at a previous scene's bowl position
would place it wrongly for the current one — but **I have not confirmed that**, and it would be
wrong to present the 0/129 as covering these scenes.

The bound this puts on the conclusion: the "no certifiable demo exists" finding rests on **direct
measurement of the destination**, which is sound for the ~90% of scenes at 0.00 cm, and is
**unproven for the ~10% whose place pose is clear**. Over 129 scenes that is roughly 13 scenes where
a demo might certify and where this run's negative result should not be trusted. Confirming or
refuting the stale-attachment hypothesis — one run, attach/detach per scene in a fresh router, or
re-run just those scenes one per process — is the first thing to do before anyone quotes 0/129 as
final.

---

# 7a closing — the 11 feasible scenes, one per process

Author: Varun Chandra, 2026-10-20. This resolves the open item above and **changes the 7a
conclusion**: the 0/129 was partly an artefact of my own harness, and two certifiable demos exist.
Artefacts: `out/place_pose_gap_all.json`, `out/one_per_process_rows.json`,
`out/one_per_proc.log`, `clips/accepted_*`.

## How many scenes are feasible at all

The place-pose clearance was scanned for all 129 scenes (scene build plus GJK, no planning):

```
TRUE payload clearance at dgcore's place pose, 129 scenes
  min 0.00   median 0.00   max 22.38 cm
  scenes >= 0.5 cm   11/129
  scenes >= 1.0 cm   11/129
  scenes >= 2.0 cm   11/129
```

The distribution is **bimodal**: either exactly 0.00 cm (118 scenes) or 9.5–22.4 cm (11 scenes).
Nothing lies between. So 118 of 129 are infeasible at any positive margin — the destination is
occupied — and exactly 11 are worth planning for. Those 11 fall on inits 18, 19 and 34.

## The 11, each in its own process with a fresh MotionGen

| scene | init | shift cm | planned | via min cm | executor | success | min clr cm | 1 cm | 2 cm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| 0:12 | 34 | 19.31 | yes | 2.83 | no | no | 0.00 | — | — | unreached `detour_v13` |
| 1:13 | 34 | 18.07 | yes | 3.05 | no | no | 0.00 | — | — | unreached `grasp` |
| 2:6 | 18 | 14.74 | yes | 5.47 | no | no | 0.00 | — | — | unreached `detour_v11` |
| 4:8 | 18 | 12.32 | yes | 5.58 | **yes** | **yes** | 0.00 | no | no | certification |
| 4:10 | 18 | 13.43 | yes | 5.53 | **yes** | **yes** | 0.00 | no | no | certification |
| 5:7 | 18 | 13.66 | yes | 5.86 | no | no | 0.00 | — | — | unreached `detour_v12` |
| **5:9** | **19** | **11.31** | **yes** | **5.77** | **yes** | **yes** | **2.93** | **YES** | **YES** | **ACCEPTED** |
| 7:19 | 18 | 16.01 | **no** | — | — | — | — | — | — | `INVALID_START_STATE_WORLD_COLLISION` |
| **8:10** | **19** | **9.97** | **yes** | **5.91** | **yes** | **yes** | **2.99** | **YES** | **YES** | **ACCEPTED** |
| 8:11 | 18 | 13.97 | yes | 5.42 | **yes** | **yes** | 0.00 | no | no | certification |
| 9:4 | 34 | 18.76 | yes | 2.92 | no | no | 0.00 | — | — | unreached `grasp` |

```
scenes 11   planned 10/11   accepted at 1 cm 2/11   accepted at 2 cm 2/11
outcomes: executor 5, certification 3, ACCEPTED 2, planner_failure 1
```

**The stale-attachment hypothesis is confirmed.** In the 129-run every one of these scenes returned
`ik_place`; one scene per process, 10 of 11 plan. A single long-lived `MotionGen` with per-scene
`attach_payload`/`detach_payload` does not reset cleanly, and the payload from a previous scene
poisons the next. That makes the previous run's **0/129 unsafe for the 11 feasible scenes**, exactly
as flagged. It does not touch the 118 scenes whose destination measures 0.00 cm.

## Two accepted demonstrations — the first in milestone 7a

| clip | init | shift | min GJK clearance | steps |
|---|---|---|---|---|
| `clips/accepted_s05a09_i19_shift11.3cm_clr2.97cm.mp4` | 19 | 11.31 cm | **3.54 cm** | 125 |
| `clips/accepted_s08a10_i19_shift10.0cm_clr2.32cm.mp4` | 19 | 9.97 cm | **2.99 cm** | 125 |

The filenames were corrected on 20 Oct to carry the scene (`shard:attempt`) as well as the init:
the first naming keyed only on the init, so the second init-19 acceptance overwrote the first and
cost a clip. The clearance in each filename (2.97 and 2.32 cm) is what the **re-render** measured;
the table's 3.54 and 2.99 cm are what the **run** measured, and they differ for the same
non-determinism noted above.

Both execute the planner's route to the end, satisfy the task's success predicate, and clear the
declared obstacles by about 3 cm — so they certify at 1 cm **and** at the project's original 2 cm.
(The 5:9 figure is 3.54 cm on the re-run against 2.93 cm in the first pass; cuRobo's trajectory
optimisation is not deterministic across processes, so the path differs slightly between runs while
the outcome does not.)

## An error of mine that these numbers depend on

`dgcore.execute` stores clearance in **centimetres** — `ep.add(obs, action, 100.0 *
rec["min_clearance"], ...)`, and `generate_demo` reads `np.min(ep.clearance)` straight into
`min_clearance_cm`. My `run_full` multiplied by 100 again. Two consequences:

- every clearance I reported from this harness was **100× too large** (the "292.86 cm" that first
  made me suspicious of the accepted demo was 2.93 cm);
- the acceptance test was `100 * clr >= 100 * FLOOR_M`, i.e. **clearance ≥ 0.01 cm, not 1 cm**.

It changed no earlier conclusion, because nearly every executed route measured exactly 0.00 and
zero fails any threshold. But it would have let a 0.5 mm demo through as "accepted at 1 cm", and
the two genuine acceptances are only trustworthy because the fixed arithmetic puts them at 2.9 and
3.5 cm. Fixed in `step4b_run.py`; the table above is recomputed with the correct units.

One scope note: this harness runs a **single rendered pass** rather than dgcore's dry pass followed
by a separate rendered replay, so the replay-reproducibility gate is not independently exercised.
The episode measured is the episode rendered.

## Place-pose renders (21 Oct)

The 0.00 cm in those 118 scenes is a claim about geometry, so it is worth seeing rather than only
tabulating. Three of them — different inits, picked from `out/place_pose_gap_all.json` — were
rebuilt on CPU with the eval-rule shift applied, the bowl moved by free-joint `qpos` to dgcore's
own nominal place pose on `plate_1` and held there with `sim.forward()` (no stepping, no planner,
no GPU), then rendered from `agentview` and from a free camera directly overhead.

| scene | init | shift | bowl → `white_place_box_1` | bowl → `bottle_of_perfume_1` | bowl x,y | `plate_1` x,y | `white_place_box_1` x,y |
|---|---|---|---|---|---|---|---|
| 0:0 | 10 | 13.30 cm | **0.000 cm** | 9.796 cm | `0.0544, 0.1531` | `0.0544, 0.1917` | `0.0272, 0.1259` |
| 5:0 | 49 | 12.33 cm | **0.000 cm** | 9.812 cm | `0.0651, 0.1738` | `0.0651, 0.2124` | `0.0110, 0.1427` |
| 4:0 | 30 | 17.18 cm | **0.000 cm** | 9.798 cm | `0.0581, 0.1669` | `0.0581, 0.2055` | `0.0303, 0.1864` |

Distances are from the same scan path as the 129-scene measurement (`DistanceEngine` hulls into
`fastgjk.gjk_distance`, padding subtracted), computed two ways — the rigid grasp→place translation
of the bowl hull, and a direct re-query of the moved state — identical to 3 dp in all three.

Six PNGs in `review/planner/placepose/`, numbers in `placepose_scan.json`. What they show: after
the eval-rule shift, `white_place_box_1` stands **on the plate**, with `bottle_of_perfume_1` on top
of it, and the bowl's place pose falls inside the box's footprint — its rim is visible emerging
from the box's lower-right face in the agent view. The bowl-to-bottle distance stays at ~9.80 cm
across all three while the shift varies 12.3–17.2 cm, because the bottle sits above the contact.

## The closing statement for 7a, corrected

> At the displacement test's eval-rule placements on oa1, **118 of 129 scenes admit no certifiable
> demonstration at any positive margin**, because the task's own place pose puts the carried bowl in
> contact with a declared obstacle: bowl-to-box clearance at the place pose is exactly 0.00 cm in
> 118 scenes and 9.5–22.4 cm in the other 11, with nothing in between. On the 11 that are
> geometrically feasible, a whole-arm planner with verified geometry plans 10 and produces **2
> certified demonstrations**, clearing the obstacles by 2.99 and 3.54 cm — passing both the 1 cm and
> the original 2 cm floor. The remaining 8 fail downstream of planning: 5 in the OSC executor and 3
> at certification with 0.00 cm.
>
> So the limiting factor is the scene, not the planner. A motion planner converts the feasible
> fraction of these placements into demonstrations at roughly a 2-in-11 rate, against the scripted
> planner's 1-in-197 over the whole set; but the feasible fraction is only 8.5% of the test's own
> placements, because the eval rule moves the obstacle stack onto the destination. Raising the yield
> means changing where the test puts the obstacle, not improving the planner.
