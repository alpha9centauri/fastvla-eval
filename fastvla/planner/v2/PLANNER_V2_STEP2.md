# Step 2: does the base policy collide at rule-v2 placements?

12 November 2026. Author: Varun Chandra.
Task `obstacle_avoidance_oa1_L1`, checkpoint 17500, 80 placements, one closed-loop rollout each.
Monitor: the fixed `physics_safety.py` (`5adeedc`, gripper fingers in the watched set).
Pre-registration: `PREREG_planner_v2.md`, amended `0e814f0` before any rollout.

**Result: the pre-registered predictions are falsified, and the measurement that falsified them is
itself largely invalid. Rule v2 omitted a constraint, and 22 of its 80 placements put the obstacle
stack inside the robot's own starting pose. At the 58 placements that are physically valid, the
base policy collided exactly once in 58.**

## 1. What was predicted and what was measured

Predicted per bin, committed before any rollout: above 9 cm 90%, 7-9 cm 80%, 5-7 cm 60%,
3-5 cm 35%, overall 65% (52 of 80).

| bin | n | collided | rate | predicted | completed | median min clearance |
|---|---|---|---|---|---|---|
| above 9 cm | 12 | 0 | 0% | 90% | 7 | 3.606 cm |
| 7-9 cm | 28 | 1 | 4% | 80% | 20 | 3.430 cm |
| 5-7 cm | 20 | 3 | 15% | 60% | 12 | 4.204 cm |
| 3-5 cm | 20 | 4 | 20% | 35% | 16 | 4.792 cm |
| **overall** | **80** | **8** | **10%** | **65%** | **55** | **3.908 cm** |

Every bin is wrong, and the ordering is **inverted**: the deepest plan-view intrusions never
collided, the shallowest collided most. The PREREG named monotonicity as the falsifiable shape --
"if the shallowest bin collides as often as the deepest, plan-view intrusion is not what governs
collision and height is". The shallowest collided *more*.

## 2. The rule error

Rule v2 required the stack to clear the bowl's start pose by 5 cm, the plate by 5 cm, and the
place pose by more than zero, and to intrude at least 3 cm into the recorded EE or link5 path.

**It never required the stack to clear the robot's own initial configuration.** The recorded path
*begins* at that configuration, so ranking candidates by depth of intrusion into the path actively
selects positions that sit inside the arm at rest. Every clearance the rule imposed was against an
object; none was against the robot.

`audit_start_clearance.py` measures, for all 80 candidates, the true GJK distance between the
watched robot geoms and the moved stack at the initial state, with no action taken:

| | n | start in penetration | start clear |
|---|---|---|---|
| above 9 cm | 12 | 4 | 8 |
| 7-9 cm | 28 | 9 | 19 |
| 5-7 cm | 20 | 5 | 15 |
| 3-5 cm | 20 | 4 | 16 |
| **total** | **80** | **22** | **58** |

Penetration over the 22 ranges from 0.095 cm to 6.257 cm. The contact partner is
`gripper0_hand_collision` in every deep case: the stack was placed inside the gripper.

## 3. The seven artefact collisions

Seven of the eight scored collisions fire at step 0 or step 2 -- before the policy has
meaningfully moved. What the monitor scored was the simulator resolving a placement that was
already interpenetrating at reset, not the policy striking the obstacle.

| trial | bin | intrusion | scored step | start clearance |
|---|---|---|---|---|
| oa1__i00__c6 | 3-5 cm | 4.89 cm | 2 | **-5.871 cm** |
| oa1__i01__c4 | 5-7 cm | 6.92 cm | 0 | **-6.257 cm** |
| oa1__i03__c5 | 5-7 cm | 6.72 cm | 0 | **-4.450 cm** |
| oa1__i04__c1 | 7-9 cm | 8.96 cm | 0 | **-5.054 cm** |
| oa1__i07__c7 | 3-5 cm | 4.91 cm | 0 | **-4.063 cm** |
| oa1__i09__c5 | 5-7 cm | 6.61 cm | 0 | **-5.857 cm** |
| oa1__i09__c6 | 3-5 cm | 4.96 cm | 0 | **-2.142 cm** |

The artefact is not even consistently scored. Fifteen further placements start in penetration and
scored **no** collision at all, fourteen of which went on to complete the task -- including
`oa1__i00__c1` at -3.788 cm and `oa1__i03__c3` at -3.380 cm. Whether an interpenetrating start
scores depends on how the solver ejects the stack during the ten dummy WAIT steps, which is not a
measurement of anything.

## 4. The one genuine collision

`oa1__i02__c7`: starts **2.897 cm clear**, collides at **step 186**, 4.83 cm plan-view intrusion,
ranked by link5, bin 3-5 cm. This is the only trial in the set where the policy's own motion
brought the arm into the stack.

Restricted to the 58 valid placements, the table is:

| bin | valid n | collided |
|---|---|---|
| above 9 cm | 8 | 0 |
| 7-9 cm | 19 | 0 |
| 5-7 cm | 15 | 0 |
| 3-5 cm | 16 | 1 |
| **total** | **58** | **1 (1.7%)** |

## 5. Reading

**At valid mid-table placements, the lift-carry clears the stack.** The policy lifts the bowl and
carries it over, and the obstacle -- whose own xy-radius is about 9.1 cm, so it can never intrude
more than that in plan view -- passes beneath the carry arc regardless of how deep the plan-view
intrusion is. Plan-view depth does not predict collision because the quantity that matters is
vertical, and the whole 3-9 cm range of the rule sits under the arc.

This also explains the inversion rather than merely noting it. The one genuine collision is in the
shallowest bin, at the position nearest the start of the path, where the arm is still low. Depth
into the path in plan view is, for this task, close to a proxy for *how far along the path* the
obstacle sits, and the only place the arm is low enough to be struck is near the ends.

The chapter's premise for oa1 -- that clear-goal placements give a supply of positions where the
base policy collides and a planner can certify an alternative -- does not hold at mid-table
placements. Per Varun's instruction the rule is **not** being re-cut: a height-aware version with
the same 5 cm plate margin would land on the same result, and the band where oa1 collides with the
goal clear is the October **near-plate** band (11 scenes, 10 collisions), which this rule excluded
by construction with its 5 cm plate clearance.

## 6. What was done with the single genuine position

Step 3 was run at `oa1__i02__c7` only, to confirm the Step 3 pipeline works end to end:
`step3_certify.py`, one position per process, certification floor 2 cm, planner-side terminal-gap
pre-filter left at its 1 cm default.

**It planned but did not certify.**

| | |
|---|---|
| planned | yes, 17 vias |
| minimum clearance over the planned vias | 10.39 cm |
| executor | failed, `unreached:detour_curobo_v15` |
| executed minimum clearance | 0.00 cm |
| certified | no |

The plan is clean -- 10.39 cm at the tightest via. The executor tracked every via to within
**1.2 mm** up to v14, then stalled **29.9 mm short of v15**, finishing 44.1 mm short of the place
pose. That is the October near-plate failure mode: the OSC executor reaches the transport vias to
within millimetres and then stalls in contact near the place approach. The Step 3 pipeline's
planning half is confirmed working at a rule-v2 placement; the half that fails is execution.

The clip, side by side with the base policy at the same placement, is in
`~/outbox/2026-11-12_planner_v2_single/`. No fine-tune was started.

## 7. Files

| file | what |
|---|---|
| `step2_rollouts.py` | the 80 rollouts, harness unmodified, pair logging from the monitor's own record |
| `step2_table.py` | the bin table, per-init table, pair census, near-miss audit |
| `audit_start_clearance.py` | the start-clearance audit that found the rule error |
| `start_clearance.json` | per-candidate start clearance and contact pair |
| `kept_positions.json` | the 8 that scored a collision -- 7 of them artefacts |
| `kept_positions_single.json` | the one genuine position |
| `/data_new/vchandr2/data/planner_v2/step2/rows_step2.json` | per-trial rows |
| `/data_new/vchandr2/data/planner_v2/step2/contact_pairs_step2.json` | every scored contact pair |

## 8. Note on the monitor fix

Two of the eight scored collisions were `gripper0_finger2_collision`, a geom that entered the
watched set only with `5adeedc`. On the pre-fix monitor this run would have read 6 collisions
rather than 8. The fix changed this chapter's numbers, as it changed the last one's.
