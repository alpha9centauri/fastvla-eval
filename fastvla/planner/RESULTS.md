# Step 4 — the planner on all 197 eval-rule oa1 scenes

Author: Varun Chandra, 2026-10-19. Milestone 7a step 4. GPU 1, one process, one warm cuRobo
instance. Artefacts: `out/step4_report.txt`, `out/step4_rows.json`, `out/step4_rows_full.json`,
`clips/`, log `PLANNER_STEP4.log`.

## Headline

**Acceptance 0/200 at 2 cm and 0/200 at 1 cm.** The scripted planner's baseline on the same scenes
is 1/197. The motion planner is not the bottleneck and never was the whole problem: it returns a
collision-free joint path on **129 of 200 scenes (64.5%)**, against 0 before step 3's fixes. What
the run establishes is that the clearance is lost *after* planning, and — measured rather than
assumed — **it is lost in the plan itself, not in the executor.**

## Scene set

200 scenes. The 2b smoke ran `gen_stagec.py` with `TAG=oa1 VARIANTS=varied N_PER=10 SHARDS=10
MAX_MULT=20 SEED=0 PLACEMENT=eval_rule` over the 50-init baseline, each shard seeded
`SEED + 131*k` with at most 20 attempts, and accepted 1 of 197. This driver mirrors that loop
serially and does **not** stop a shard on acceptance, so all 10 shards run their full 20 draws:
**197 scenes are the identical draws the smoke made**, and 3 are the continuation of shard 4, the
only shard that accepted and therefore stopped short. Acceptance is reported over both.

Realised obstacle shift over the 200 scenes: 6.97–20.89 cm, mean 15.45 cm.

## Acceptance

| certification floor | all 200 | the 2b smoke's 197 |
|---|---|---|
| 2 cm (the project's floor) | **0/200 = 0.0%** | 0/197 = 0.0% |
| 1 cm | **0/200 = 0.0%** | 0/197 = 0.0% |
| scripted planner, 2 cm | — | 1/197 = 0.5% |

Both floors come from one run: certification ran at 1 cm and every executed demo's own minimum GJK
clearance was recorded, so the 2 cm number is recovered by counting demos that also clear 2 cm. The
2 cm standard was not weakened; it was applied afterwards to the same measurements.

## Planning time

Scenes where `plan_single` returned a path (n=129):

| | mean | median | p90 | range |
|---|---|---|---|---|
| **single successful call** | 0.91 s | 0.82 s | 1.05 s | 0.58–3.45 s |
| **total across candidates, per scene** | 5.40 s | 3.72 s | 10.17 s | 0.63–36.23 s |

Hand-off candidates tried per scene: mean 15.2 of 24, of which collision-aware IK accepted a mean
of 4.14. **Every one of the 200 scenes had at least one hand-off accepted by IK**, so there is no
"no clear hand-off" bucket at all — the wider sweep solved that failure mode outright.

Whole scene, including the dry pass and the replay: mean 52.2 s, median 56.9 s, range 13.9–117.5 s.
Total wall clock 174.1 min for 200 scenes.

## Hand-off height and tilt

No demo was accepted, so these are the distributions over the 129 scenes that produced a path, and
they are labelled as such rather than presented as accepted-demo statistics.

| hand-off height | n | | horizontal offset | n |
|---|---|---|---|---|
| 8 cm | 4 (3.1%) | | 3 cm | 11 (8.5%) |
| **12 cm** | **54 (41.9%)** | | **6 cm** | **48 (37.2%)** |
| **16 cm** | **59 (45.7%)** | | **9 cm** | **70 (54.3%)** |
| 20 cm | 1 (0.8%) | | | |
| 24 cm | 9 (7.0%) | | | |
| 28 cm | 2 (1.6%) | | | |

"Lowest accepted wins" is biting correctly on height — 88% of scenes take 12 or 16 cm — but the
offset piles up at the far end of the sweep (54% at 9 cm, the maximum offered), which says the
offset range is the binding one and should be extended before the height range is.

**Wrist tilt: mean 24.6°, median 24.8°, p90 31.9°, range 9.6–36.5°, and 0 of 129 exceed 45°.**
Raising the limit from 20° to 45° removed that failure mode entirely: at 20° it would have rejected
a substantial share (the median alone is above it), and at 45° it rejects nothing. The empty-bowl
reasoning holds — no plan comes close to inverting the payload.

## Failure breakdown

| cause | n | % |
|---|---|---|
| executor — a waypoint was never reached inside its step budget | **120** | 60.0% |
| plan fail — IK accepted a hand-off, `plan_single` found no path | **71** | 35.5% |
| certification — the route ran to completion, GJK clearance below the floor | **9** | 4.5% |
| no clear hand-off | 0 | 0% |
| tilt above the limit | 0 | 0% |

## The finding that matters, and a correction to my own status update

Mid-run I told you the bottleneck had "moved out of the planner and into the executor", on the
strength of the 60% executor bucket. **That reading was incomplete and in one respect wrong.**

First, the 9 "certification" scenes did not, as I said, "run to completion and drift into the
bottle" — they recorded a dry-pass clearance of exactly 0.00 cm and produced no episode at all.

Second, and decisively: **all 129 scenes that produced a path have a dry-pass minimum GJK clearance
of either 0.000 or 0.072 cm.** Not one reaches 1 cm. A distribution with two values across 129
scenes is not an executor-tracking distribution; it is contact, every time.

So the executed clearance was compared against the clearance at the **planned** vias
(payload hull carried rigidly from the grasp pose, GJK against the obstacle's 244 pieces through
the labeler's own engine). On the cheapest reproducible scene, s09a00, init 25, 11.47 cm shift,
plan reported clear:

```
payload-vs-obstacle GJK clearance AT THE PLANNED VIAS (24 vias)
  min 0.39 cm   median 7.76 cm   max 30.31 cm
  vias under 2 cm: 3/24      vias under 1 cm: 2/24
```

**The plan itself is not 2 cm clear.** cuRobo called that path collision-free while the certifier
measures 0.39 cm at its tightest via. So the clearance is lost in planning, not in execution, and
the executor is a second and separate problem rather than the explanation.

Why the plan is optimistic, and why this is the opposite of step 2's finding: step 2 measured
cuRobo **more conservative** than GJK by 0.7–3.8 cm, but that was for the *arm*, whose links are
sphere-fitted to the Panda meshes. The *attached payload* is represented by surface-sampled spheres
of radius 0.005 m (`SAMPLE_SURFACE`, chosen in step 3 because cuRobo's volume fit welded a solid
11×11×5 cm cuboid to the gripper and refused everything). Surface sampling at 5 mm under-represents
the bowl against its exact hull, so for the payload cuRobo is **optimistic** where for the arm it is
conservative. Step 3 traded one error for its mirror image and I did not check the direction.

Caveat on strength: the planned-via measurement is **n=1 scene**. It is strongly consistent with all
129 scenes landing at 0.00–0.072 cm, but it is one scene, and the honest next step is to run it
across the 129 before acting on it.

## What follows from this

1. **Plan against inflated obstacles.** Nothing currently requires the planner to clear 2 cm; it is
   asked only for "not in collision". Inflating the obstacle meshes by the certification floor
   before planning makes a cuRobo-clear plan GJK-clear by construction, and is the one change most
   likely to convert the existing 129 paths into certifiable demos.
2. **Fix the payload representation's direction of error.** Either a larger surface-sphere radius or
   a volume fit that does not degenerate to a cuboid, chosen by measuring payload clearance against
   GJK rather than by whether candidates pass.
3. **Then the executor.** 120 of 129 routes still fail to reach a waypoint. The vias are a stride-3
   thinning of cuRobo's interpolated path, and the executor drives straight lines between them with
   a 16 mm arrival tolerance; both the thinning and the tolerance are suspects, and neither has been
   measured yet.

None of these were done. They are step-5-or-later decisions and the brief says stop here.

## Clips

`fastvla/planner/clips/` — three scenes, rendered with the cameras on through dgcore's own executor
and monitor. **Nothing was accepted and no scene produced an episode**, so `generate_demo` had no
frames to give; these are the three closest-to-accepted (all from the `certification` tier, the
only scenes where the executor ran the whole route), rebuilt and executed for the render. Each
`.json` carries the scene's identity, which was checked against the recorded init and shift before
rendering.

| clip | init | shift | hand-off | tilt | executor | min GJK clearance |
|---|---|---|---|---|---|---|
| `step4_s09a00_i25_shift11cm_certification.mp4` | 25 | 11.47 cm | 16/9 cm | 22.3° | **failed** at `pre_place` after 192 steps | 0.00 cm |
| `step4_s02a06_i18_shift15cm_certification.mp4` | 18 | 14.74 cm | 16/9 cm | 25.3° | **ok**, 229 steps | 0.00 cm |
| `step4_s05a07_i18_shift14cm_certification.mp4` | 18 | 13.66 cm | 16/9 cm | 23.5° | **ok**, 234 steps | 0.00 cm |

Two of the three executed the planned route to the end and still touched the bottle — which is the
same conclusion the planned-via measurement reached, visible rather than tabulated.
