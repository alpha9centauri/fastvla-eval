# The runtime collision monitor

A linear probe on the policy's own activations, read at every forward, turned into an
alarm score the server can act on. Everything here is measured on this harness; the
numbers come from `bench/out/m7b_probe.json` and the run directories named in each table.

## 1. What is captured, and what it costs

The probe reads the hidden state at the reference probe's exact site: the output of the
language model's final norm, averaged over the 56 action-token positions, cast to float16
(`capture.py`). It is a forward hook, so it sees what the policy computed and changes
nothing about it.

| | |
|---|---|
| capture cost | **0.0077 ms per forward**, 0.004% of a 204.9 ms forward |
| storage | **0.289 MB per rollout** on disk (compressed npz, all arrays), of which 0.286 MB is the activations |
| per forward | 8192 bytes (4096 float16); ~35 forwards in a 300-step rollout |

The hook refuses `MAX_BATCH != 1`. The reference hook indexes `h[0, ...]` (assumption B2),
so at batch > 1 every request in a batch would be handed the first one's hidden state.

**The capture changes no result.** The 60-start acceptance set run with capture on gives
29 collisions, 25 completions, zero collision flips, zero completion flips, max per-start
clearance difference 0.0000 cm, at 539.3 rollouts/hour
(`bench/out/m4_accept_M7B_CAPTURE_ON.json`).

## 2. Train and test

| set | manifest | rollouts | forwards | scenes |
|---|---|---|---|---|
| train | `train_unshifted` (inits 10-49) | 240 | 8081 | 240 |
| test, unshifted | `test_unshifted` (inits 0-9) | 60 | 2040 | 60 |
| test, intrusion-5 | `intr5` | 48 | 1787 | 48 |

The split is by scene and the sets are disjoint; the fit asserts it. `test_unshifted` is
byte-identical to `ctrl60`, the acceptance set, so the false-alarm numbers below and the
parity check in section 6 are measured on the same 60 starts.

**Correction to the milestone's plan.** It asked for "intrusion-5 where feasible" in the
*training* set as well. That set is empty: `out/feasible_set.json` covers inits 0-9 only,
so there are no intrusion-5 placements for inits 10-49. Training is therefore
unshifted-only and **every intrusion-5 scene is held out** — which makes the intrusion-5
column a distribution shift the probe never saw, rather than a weaker in-domain result.

## 3. The probe

Ridge from the activation to log10 clearance in metres, floored at 1e-4 m — the
reference's target and the reference's closed form (`probe.py: fit_ridge`), fitted on
pre-contact forwards.

Alpha by leave-scene-out 5-fold CV **on the training set only**:

| alpha | 1e1 | 3e1 | 1e2 | **3e2** | 1e3 | 3e3 | 1e4 | 3e4 | 1e5 | 3e5 | 1e6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out Spearman | +0.813 | +0.855 | +0.882 | **+0.892** | +0.890 | +0.876 | +0.844 | +0.800 | +0.745 | +0.690 | +0.580 |

The first grid started at 1e3 and the optimum landed on its edge, which is not an optimum
but an unexplored direction; extended downwards, the optimum is interior at **3e2**. The
reference's fixed 1e5 is in the grid and is worse here (+0.745), on a training set 4x
larger than the one it was chosen for.

## 4. Test results

The alarm score is **minus** the predicted log clearance, so higher means closer and one
`>=` is the alarm.

| | unshifted (60) | intrusion-5 (48) |
|---|---|---|
| Spearman to true clearance, pre-contact forwards | **+0.868** | **+0.480** |
| Spearman, all forwards | +0.663 | +0.162 |
| AUROC, contact within 8 control steps | **0.918** | 0.743 |
| AUROC, within 16 | 0.807 | 0.670 |
| AUROC, within 32 | 0.756 | 0.630 |
| pre-contact forwards / positives at k=16 | 1313 / 55 | 740 / 77 |

AUROC falls as the horizon grows, which is the right direction: the score does rise into
contact rather than merely labelling a scene. For comparison, the reference's offline
probe reports +0.5639 held-out Spearman under leave-one-init-out on 1313 pre-contact
samples from the 60 inits; this one is fitted on 240 disjoint scenes and tested on those
60, and gets +0.868 — and +0.480 on scenes whose geometry it has never seen.

### Lead time

Control steps between the first alarm and first contact, over every colliding rollout, at
the operating point in section 5.

| | unshifted | intrusion-5 |
|---|---|---|
| colliding rollouts | 29 | 40 |
| alarmed before contact | **28** | **38** |
| no alarm before contact | 1 | 2 |
| median | 95.0 | 45.5 |
| p10 / p90 | 11.6 / 122.5 | 8.0 / 105.1 |
| min / max | 6 / 188 | 2 / 156 |

A median lead of 95 control steps is not 95 steps of warning. It is the honest reading of
the next section: at this threshold the score is above the line for most of most
rollouts, so the "first alarm" is usually an early nuisance alarm rather than a prediction
of the contact that eventually happens.

## 5. False alarms, and the operating point

Threshold chosen for **90% recall at k=16 on the training set** (+1.0178), never on test,
so the test false-alarm rate is not fitted on the set it is reported for. Measured over
rollouts that never contact:

| | unshifted | intrusion-5 |
|---|---|---|
| clean rollouts | 31 | 8 |
| clean rollouts raising at least one alarm | **31 of 31** | **8 of 8** |
| alarms per 100 control steps | **6.54** | 7.75 |

**Every clean rollout alarms.** That is the headline limitation and it is not a bug in the
threshold: the sweep below shows the whole trade-off, with the threshold always chosen on
train and measured on test.

| train recall target | threshold | test recall at k=16 | clean rollouts alarming | alarms / 100 steps |
|---|---|---|---|---|
| 0.50 | +1.371 | 0.509 | 16 of 31 | **0.68** |
| 0.60 | +1.298 | 0.636 | 22 of 31 | 1.31 |
| 0.70 | +1.214 | 0.691 | 25 of 31 | 2.49 |
| 0.80 | +1.112 | 0.818 | 30 of 31 | 4.35 |
| **0.90** | **+1.018** | **0.855** | **31 of 31** | **6.54** |
| 0.95 | +0.920 | 0.909 | 31 of 31 | 8.23 |

(unshifted; the intrusion-5 sweep is in `bench/out/m7b_probe.json`.)

Read down the last two columns: nine tenths of the alarm budget buys the last third of
the recall. At 50% recall the monitor is quiet — 0.68 alarms per 100 steps, half the clean
rollouts silent — and still catches half the contacts within 16 steps. The milestone fixed
the operating point at 90% recall and that is what sections 6 and 7 use, but the knee is
at 0.6-0.7 and a monitor meant to be lived with belongs there.

## 6. The figure

![alarm score against time](alarm_score.png)

Three held-out rollouts, one per panel: a collision, a near miss, a clean completion. The
contact step is marked in red. The collision panel is the argument and the counter-argument
at once — a clear, sharp rise into contact at step 196, and a first crossing at step 8 that
is credited as "188 steps of lead" but is plainly a nuisance alarm.

## 7. Run-to-run reproducibility (measured, and it bounds what section 8 can claim)

Two runs of the **same manifest in the same configuration** are not always trace-identical:

| pair | both capture | traces differing | max clearance diff | collision flips | completion flips |
|---|---|---|---|---|---|
| `ctrl60_n8b1` vs `m4_t1_n8b1` | off | 4 of 60 | 5.3e-3 m | 0 | 0 |
| `m4_t1_n8b1` vs `m7b_test_unshifted` | off vs on | 2 of 60 | 5.3e-3 m | 0 | 0 |
| `ctrl60_n16b1` vs `m7b_test_unshifted` | off vs on | 0 of 60 | 0 | 0 | 0 |
| `m7b_intr5_gate` vs `m7b_test_intr5` | off vs on | 4 of 48 | 7.2e-2 m | 0 | **1** |

Two facts follow. It is **not** the capture hook: two capture-off runs differ from each
other as much as any other pair. And in every one of the 8 divergent traces the first
difference falls inside the first control step of the first differing forward — the
simulator is deterministic, the policy forward is not bit-reproducible run to run at
`TORCH_DETERMINISTIC=0`, which is the configuration all the baselines were measured in
(M4 measured determinism mode at 526.2 rollouts/hour, -2.9%, and rejected it;
`fastvla/server/FORWARD.md` step 0). The divergences are 1e-4 to 4e-3 in action units and
amplify only where the dynamics are already marginal.

So the noise floor for a per-start comparison on this harness is **0 collision flips** and
**0-1 completion flips**, and any effect claimed below has to clear it.

## 8. The monitor in the server

`MONITOR=off|log|stop`, off by default, refused at `MAX_BATCH != 1` for the same reason
the capture is. It runs between the forward and the reply, so `stop` brakes the very chunk
whose activation raised the alarm rather than the next one.

- **`log`** scores every forward, puts the score in the response header (the worker writes
  it into the rollout record as `monitor_scores`) and into the Prometheus metrics
  (`fastvla_monitor_scored_total`, `_alarms_total`, `_alarm_rate`, `_held_rollouts`,
  `_last_score`). Actions are untouched.
- **`stop`** latches per rollout: once the score crosses the threshold, every chunk for
  that rollout has its pose delta zeroed and its gripper command passed through unchanged.
  The zeroing is applied to the raw chunk before the worker's `process_action`, so the
  gripper goes through normalise/invert exactly as it otherwise would.

The activation is shipped to the worker only when a collection run asks for it
(`CAPTURE_ACTIVATIONS=1`); the monitor reads it in the server, so `log` pays no 8 KB frame.

### Cost (`bench/out/m7b_monitor_cost.json`)

| run | mode | rollouts/hour | forward ms | capture ms | monitor ms | alarms per forward |
|---|---|---|---|---|---|---|
| `m4_t1_n8b1` | off | **544.8** | 184.3 | — | — | — |
| `m7b_test_unshifted` | off | 539.3 | 185.6 | 0.0050 | — | — |
| `m7b_log_ctrl60` | **log** | **546.2** | 183.4 | 0.0054 | **0.0616** | 0.617 |
| `m7b_stop_ctrl60` | stop | 484.8 | 184.9 | 0.0047 | 0.0629 | 0.830 |
| `m7b_test_intr5` | off | 488.9 | 185.4 | 0.0049 | — | — |
| `m7b_stop_intr5` | stop | 474.2 | 184.7 | 0.0050 | 0.0617 | 0.866 |

**`MONITOR=log` costs 0.067 ms per forward** — 0.0054 capture plus 0.0616 score — which is
**0.037% of a 183.4 ms forward**. End to end it is **546.2 rollouts/hour against the 544.8
baseline, +0.26%**: within noise, and in the direction that proves the point, since a
0.067 ms addition to a 184 ms forward cannot make anything faster.

`stop` is slower (484.8) and none of that is the monitor: its per-forward cost is the same
0.063 ms. A braked rollout never completes, so it runs to the 300-step cap instead of
ending early — mean rollout length 298 steps against 275 unbraked. More steps per rollout,
same speed per step.

## 9. What stopping does

`MONITOR=stop` at the section 5 operating point, per start, against the same manifest run
with `MONITOR=off`. Both matrices are in `bench/out/m7b_stop_{ctrl60,intr5}.json`.

### Unshifted, 60 starts (`m7b_test_unshifted` → `m7b_stop_ctrl60`)

collisions **29 → 1**, completions **25 → 1**, 60 of 60 rollouts braked.

| collision | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 1 | **28** |
| **off: no** | **0** | 31 |

| completion | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 1 | **24** |
| **off: no** | 0 | 35 |

### Intrusion-5, 48 starts (`m7b_test_intr5` → `m7b_stop_intr5`)

collisions **40 → 2**, completions **4 → 0**, 48 of 48 rollouts braked.

| collision | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 2 | **38** |
| **off: no** | **0** | 8 |

| completion | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 0 | **4** |
| **off: no** | 0 | 44 |

Zero newly-colliding starts in either set, so the brake never made a start worse. It also
braked every single rollout, at a median first-hold of control step **8** (unshifted;
p90 146) and step **12** (intrusion-5; p90 106) — out of 300.

**This is a brake, not an avoidance.** It holds the arm where it is; it has no notion of
which direction is safe, and it never resumes. At this operating point it brakes
essentially at the start of every rollout, which is why it removes 66 of 69 collisions and
28 of 29 completions at the same time. The honest summary of the 60-start result is not
"collisions fell 29 → 1" but "the arm stopped moving, so almost nothing happened at all" —
and the 28 completions lost are the price. Both numbers come from one cause, and reporting
the first without the second would be a lie of omission.

For what a *useful* operating point looks like, section 5's sweep is the place to start:
at 50% train recall only 16 of 31 clean rollouts alarm at all, so a `stop` run there would
leave roughly half the clean rollouts untouched. That run is not in this milestone.

For context, the research tree's probe-gated steering (not re-run here, and a different
intervention — it steers rather than brakes) reports 14/60 collisions and 18/60
completions against the 29/25 zero-arm baseline: 0 rescued, 3 lost.

## 10. Reading the artefacts

| file | what |
|---|---|
| `bench/out/m7b_probe.json` | every number in sections 3-5, including both recall sweeps |
| `bench/out/m7b_probe.npz` | the fitted probe: `w`, `mu`, `sd`, `ybar`, `alpha`, `threshold` |
| `bench/out/m7b_intr5_gate.json` | the step-1 gate against the 17500 reference |
| `bench/out/m7b_monitor_cost.json` | section 8's table |
| `bench/out/m7b_stop_{ctrl60,intr5}.json` | section 9's matrices, with the per-start keys |
| `bench/out/m4_accept_M7B_MONITOR_LOG.json` | the parity proof in section 11 |
| `bench/runs/<id>/arrays/<key>.npz` | per rollout: `monitor_scores`, `monitor_held`, `activations`, `activation_steps`, beside `clearance` and `contact` |

One field is not what its name would suggest: the rollout row's
`monitor_n_held_forwards` counts forwards that were *held*, not alarms — the worker is
never told the threshold, so it cannot count alarms, and in `log` mode the value is 0 even
while the monitor alarms on 62% of forwards. Alarm counts in this document are recomputed
from the `monitor_scores` array against the probe's threshold
(`bench/m7b_monitor_report.py`). The field was called `monitor_n_alarms` during the runs
above, so their artefacts carry the old name; it is written after a rollout ends and
touches no action.

## 11. Parity: `MONITOR=log` changes nothing

`bench/accept.py` on `m7b_log_ctrl60`, the 60-start acceptance set with `MONITOR=log`:

| | |
|---|---|
| collisions | **29** (required 29) |
| completions | **25** (required 25 or 26) |
| collision flips vs the M3 N=16 reference | **0** |
| completion flips | **0** |
| transition matrix | **60/60 diagonal** |
| max per-start clearance difference | **0.0000 cm** |
| rollouts/hour | 546.2 |
| verdict | **PASS** |

There are two reasons to believe it, and the second is the stronger one. The run passes
the gate. And it *must* pass, by construction: in `log` mode the monitor is a read of a
forward hook's output followed by a dot product, after the forward has returned and
outside the action path. The only thing it writes is a float in the response header. The
capture hook it depends on was separately shown not to change results — section 1, and
the N=16 reference comparison in section 7, where the capture-on run is trace-identical
over all 60 rollouts.
