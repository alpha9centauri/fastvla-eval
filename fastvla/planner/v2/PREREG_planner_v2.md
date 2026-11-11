# Placement rule v2 — pre-registration

Author: Varun Chandra. **Written 11 November 2026, before any rollout of step 2 was run.**
Chapter: planner at clear-goal placements (oa1). Measurement only; no fine-tuning.

## Why a new rule

The old intrusion test moved the obstacle stack onto the bowl's destination, so on 118 of 129 oa1
scenes the place pose put the carried bowl in contact with a declared obstacle at exactly 0.00 cm
and **no certifiable demonstration exists at any positive margin** (README Part 4,
`RESULTS_4B.md`). On the 11 geometrically feasible scenes the planner planned 10 and certified 2.

The limit was the scene, not the search. Rule v2 keeps the goal clear and puts the obstacle on the
policy's own carry path instead, which is where `DATA_CHAPTER_VERDICT.txt` says the question
actually lives.

## The rule

For each of the 10 standard oa1 inits, take the base policy's recorded carry path — grasp point to
place pose — from the saved control-arm trajectory. Candidate obstacle positions are a **2 cm
grid** over the table region between the bowl's start and the plate. A candidate survives only if:

1. the perfume stack's footprint is **at least 5 cm** from the bowl's start footprint;
2. the stack's footprint is **at least 5 cm** from the plate's edge;
3. the bowl's **place pose does not intersect the stack** — the goal stays clear, which is the
   whole point of the chapter.

Survivors are ranked by how deep the stack intrudes into the recorded path, measured as the
nearest-point distance, and **up to 8 per init are kept with intrusion of at least 3 cm**.

Footprints are hull xy-radii from the certifier's own geometry (`SceneGeom`), so the rule and the
thing that certifies agree about size.

## Two paths, not one — the choice and its reason

Distance is taken to **both** the end-effector path and the **`robot0_link5`** path, and the
ranking uses the **smaller of the two**. Every candidate records `ranked_by` ∈ {`ee`, `link5`}.

`DATA_CHAPTER_VERDICT.txt` states that on oa1 the obstacle has to sit on the plate approach, "the
segment where the arm descends to release the bowl and where **robot0_link5, not the end effector,
is what fouls the bottle**". Ranking on the end effector alone would therefore systematically
mis-site the obstacle relative to the link that actually collides. The same link was independently
named last month as the actor behind `oa4/i09`'s scored collision in the Test 3d sweep.

The link5 path is not in the saved trajectories. It was extracted by one CPU replay pass over the
10 recorded control-arm action sequences (`v2/extract_link5.py`), saved beside the originals as
`out_onp_control/traj_link5_onpcontrol.npz`. The replay reproduces the recorded end-effector path
to **0.55–0.60 mm**, which is the check that it is the same trajectory.

**It changes the answer on 31 of 80 candidates (39%)**, where link5 is strictly closer to the
stack than the end effector is. Over the kept set the end effector's signed distance runs −9.15 to
+3.09 cm while link5's runs −9.15 to +6.15 cm, so the two parts genuinely disagree about where the
arm passes.

## A stated limitation of the measure

Intrusion is a **plan-view (x, y)** nearest-point distance, as "nearest-point distance" implies for
a stack standing on the table. It does not model height. A candidate the arm passes cleanly *over*
still scores as deeply intruding. Whether that matters is exactly what step 2 measures, and it is
the main reason the prediction below is not higher.

## Counts per init

| init | grid | survivors | kept | ranked by `ee` | by `link5` | min intrusion cm | max intrusion cm |
|---|---|---|---|---|---|---|---|
| 0 | 1056 | 500 | 8 | 5 | 3 | 8.46 | 9.10 |
| 1 | 1054 | 492 | 8 | 4 | 4 | 8.39 | 9.12 |
| 2 | 1024 | 468 | 8 | 5 | 3 | 8.18 | 9.13 |
| 3 | 992 | 429 | 8 | 5 | 3 | 8.11 | 8.89 |
| 4 | 1054 | 490 | 8 | 4 | 4 | 8.22 | 9.10 |
| 5 | 1088 | 515 | 8 | 4 | 4 | 8.37 | 8.90 |
| 6 | 990 | 430 | 8 | 6 | 2 | 8.15 | 9.15 |
| 7 | 992 | 437 | 8 | 6 | 2 | 8.13 | 9.15 |
| 8 | 1024 | 450 | 8 | 5 | 3 | 8.18 | 9.05 |
| 9 | 992 | 425 | 8 | 5 | 3 | 7.79 | 9.11 |
| **total** | **10266** | **4636** | **80** | **49** | **31** | **7.79** | **9.15** |

Every init yields the full 8. Intrusion is tightly clustered at 7.79–9.15 cm (median 8.71) because
the top 8 of several hundred survivors are kept: the stack centre sits within about a centimetre of
the path in plan view.

## Prediction, written before step 2 runs

**I predict the base policy collides at 70% of the 80 candidates** — roughly 56, with a range I
would not be surprised by of 50–65%.

The reasoning, so the error is readable afterwards. For: these placements are the deepest
intrusions available, with the stack centre essentially on the path in plan view, and the old
generator's 0-of-30 collision rate was because its placements were *off* the path — the cosine
between the two rules' shifts had a median of 0.087. Against: intrusion is plan-view only, and
link5 rides at z 1.16–1.49 m while the end effector descends to 0.97 m at the place pose, so a
stack whose top sits near 1.2 m can be passed over on the approach and only fouled on the descent.
Candidates ranked by `ee` are the ones most likely to be passed over, and they are 49 of the 80.

If the rate comes in far below 50%, the plan-view measure is the thing to fix — by scoring
intrusion against the path in 3-D, or by filtering on the stack's height against the path's height
at the nearest point.

## What step 2 does with this

Checkpoint 17500 runs once at each of the 80 kept candidates. Collision, completion, min clearance
and the **named contact pair** are recorded. Only positions where the base policy collides go
forward to step 3, because a position the policy already handles cannot demonstrate an avoidance
it does not already perform — the error the data chapter made and recorded in its own verdict.

Candidate list, with per-candidate `x`, `y`, `shift`, `intrude_cm`, `d_ee_cm`, `d_l5_cm` and
`ranked_by`: `v2/candidates_v2.json`.
