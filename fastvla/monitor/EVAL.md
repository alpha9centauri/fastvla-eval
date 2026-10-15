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

Research-tree paths in this document (`out/...`, `out_instruct/...`, `docs/RESULTS.md`)
are relative to `/data_new/vchandr2/vla-safety-interp/src/displacement`, the `DISP`
directory the harness imports from -- not to the repository root.

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
| alarmed before contact | **18** | **12** |
| no alarm before contact | **11** | **28** |
| median | 74.0 | 78.0 |
| p10 / p90 | 4.1 / 90.9 | 30.0 / 125.5 |
| min / max | 1 / 102 | 1 / 148 |

Read this with the row above it, not on its own. A median lead of 74 control steps is not
74 steps of reliable warning: 11 of 29 unshifted collisions and 28 of 40 shifted ones
raise no alarm before contact at all, and the p10 of 4.1 steps says a tenth of the alarms
that do fire arrive with half a chunk to spare. At the 90% point of section 5 almost every
collision is alarmed (28 of 29, 38 of 40) at a longer median, but then so is every clean
rollout — the lead time only means something alongside the false-alarm rate it was bought
with.

## 5. The operating point

Two things define it: a **threshold** (+1.2979) and a **debounce of 2** — the monitor acts
only when the threshold is crossed on two consecutive forwards for the same rollout. The
threshold is the one that gives **60% recall at k=16 on the training set**, never chosen on
test, so no test false-alarm number is fitted on the set it is reported for. Both values
travel inside `m7b_probe.npz`, so the server's default is the point this document reports.

![recall against false alarms](operating_curve.png)

**Why 60% and not 90%.** On the unshifted curve at debounce 2, going from 50% to 60%
recall buys 4 more collisions caught for 0.48 more alarms per 100 steps — 8.3 collisions
per unit of alarm. From 60% to 70% it buys 2 for 1.02 (2.0 per unit), and it stays flat
after that. The bend is at 60%.

**Why a debounce.** A single forward over the line is one sample of a noisy score. Two in
a row costs almost nothing in recall at a fixed threshold and removes the isolated
crossings: at the chosen threshold it takes the unshifted false-alarm rate from 1.31 to
0.81 per 100 steps and the clean rollouts that alarm at all from 22 to 18 of 31.

### What it costs and catches, against the 90% point

| | unshifted, chosen | unshifted, 90% | intrusion-5, chosen | intrusion-5, 90% |
|---|---|---|---|---|
| threshold / debounce | **+1.2979 / 2** | +1.0178 / 1 | **+1.2979 / 2** | +1.0178 / 1 |
| recall at k=16 | 0.291 | 0.855 | 0.182 | 0.779 |
| collisions alarmed before contact | **18 of 29** | 28 of 29 | **12 of 40** | 38 of 40 |
| clean rollouts raising any alarm | **18 of 31** | 31 of 31 | **7 of 8** | 8 of 8 |
| **false alarms per 100 control steps** | **0.81** | 6.54 | **1.69** | 7.75 |
| alarming forwards / control steps | 61 / 7543 | 493 / 7543 | 40 / 2360 | 183 / 2360 |
| lead time, median | 74.0 | 95.0 | 78.0 | 45.5 |
| lead time, p10 / p90 | 4.1 / 90.9 | 11.6 / 122.5 | 30.0 / 125.5 | 8.0 / 105.1 |

The chosen point gives up most of the recall — 18 of 29 unshifted collisions alarmed
instead of 28, and only 12 of 40 on the shifted scenes — to cut the false-alarm rate by
8x and leave 13 of 31 clean rollouts completely silent. The 90% point leaves none silent,
which is what made it useless as a brake (section 9).

### The full sweep

Threshold always chosen on train, measured on test. `d1` is one crossing, `d2` the chosen
debounce.

| train recall | threshold | d1: clean alarming | d1: per 100 | d1: caught | d2: clean alarming | d2: per 100 | d2: caught |
|---|---|---|---|---|---|---|---|
| 0.50 | +1.371 | 16 of 31 | 0.68 | 24 of 29 | 11 of 31 | **0.33** | 14 of 29 |
| **0.60** | **+1.298** | 22 of 31 | 1.31 | 26 of 29 | **18 of 31** | **0.81** | **18 of 29** |
| 0.70 | +1.214 | 25 of 31 | 2.49 | 27 of 29 | 23 of 31 | 1.83 | 20 of 29 |
| 0.80 | +1.112 | 30 of 31 | 4.35 | 28 of 29 | 27 of 31 | 3.49 | 23 of 29 |
| 0.90 | +1.018 | 31 of 31 | 6.54 | 28 of 29 | 31 of 31 | 5.69 | 25 of 29 |
| 0.95 | +0.920 | 31 of 31 | 8.23 | 29 of 29 | 31 of 31 | 7.28 | 26 of 29 |

(unshifted; the intrusion-5 sweep is in `bench/out/m7b_probe.json` and the right-hand panel
of the figure above.)

## 6. The figure

![alarm score against time](alarm_score.png)

Three held-out rollouts, one per panel: a collision, a near miss, a clean completion. The
contact step is marked in red, and the dashed line is the chosen threshold with its
debounce applied, so what is drawn is what the monitor acts on.

The collision panel is a **chosen** example, not a typical one — the latest-contact
collision among those the monitor catches — because the panel is there to show the
mechanism. At this operating point 11 of 29 unshifted collisions raise no alarm before
contact at all, which is the table in section 5 and the real cost of the quieter
threshold. The near miss and the clean completion are the extremes of their classes
(tightest approach, widest approach) and both stay under the line for the whole rollout.

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
  (`fastvla_monitor_scored_total`, `_crossings_total`, `_alarms_total`, `_alarm_rate`,
  `_held_rollouts`, `_last_score` — crossings and alarms differ by the debounce). Actions
  are untouched.
- **`stop`** latches per rollout: once the score has crossed the threshold on `debounce`
  consecutive forwards for that rollout, every chunk for it has its pose delta zeroed and
  its gripper command passed through unchanged. The debounce count is per rollout and
  resets on any forward under the threshold; `MONITOR_DEBOUNCE` overrides the value stored
  in the probe. The same definition is used offline (`probe.debounce_mask`) and online
  (`RuntimeMonitor`), and a test asserts the two agree on a random crossing pattern, so
  the number this document reports and the number the server acts on cannot drift apart.
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
| `m7b_stop_ctrl60` | stop (90%) | 484.8 | 184.9 | 0.0047 | 0.0629 | 0.830 |
| `m7b_stopknee_ctrl60` | **stop (chosen)** | 518.6 | 183.9 | 0.0047 | 0.0650 | 0.550 |
| `m7b_test_intr5` | off | 488.9 | 185.4 | 0.0049 | — | — |
| `m7b_stop_intr5` | stop (90%) | 474.2 | 184.7 | 0.0050 | 0.0617 | 0.866 |
| `m7b_stopknee_intr5` | **stop (chosen)** | 474.1 | 184.9 | 0.0047 | 0.0631 | 0.516 |

**`MONITOR=log` costs 0.067 ms per forward** — 0.0054 capture plus 0.0616 score — which is
**0.037% of a 183.4 ms forward**. End to end it is **546.2 rollouts/hour against the 544.8
baseline, +0.26%**: within noise, and in the direction that proves the point, since a
0.067 ms addition to a 184 ms forward cannot make anything faster.

`stop` is slower and none of that is the monitor: its per-forward cost is the same
0.063 ms in every mode. A braked rollout never completes, so it runs to the 300-step cap
instead of ending early. That is why the ordering tracks how much braking happened rather
than anything about cost — 484.8 rollouts/hour when all 60 were braked from step 8, 518.6
when 46 were braked from step 92. More steps per rollout, same speed per step.

The alarm column is the debounced alarm, which is what the monitor acts on. At the chosen
point the raw threshold is crossed on 0.580 of forwards and the debounce reduces that to
0.550; `bench/out/m7b_monitor_cost.json` carries both.

## 9. What stopping does

`MONITOR=stop` at the section 5 operating point (threshold +1.2979, debounce 2), per
start, against the same manifest run with `MONITOR=off`. Matrices in
`bench/out/m7b_stopknee_{ctrl60,intr5}.json`.

### Unshifted, 60 starts (`m7b_test_unshifted` → `m7b_stopknee_ctrl60`)

collisions **29 → 11**, completions **25 → 12**, **46 of 60** rollouts braked, median
first hold at control step **92** (p10 16, p90 184, min 8, max 272).

| collision | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 11 | **18** |
| **off: no** | **0** | 31 |

| completion | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 12 | **13** |
| **off: no** | 0 | 35 |

### Intrusion-5, 48 starts (`m7b_test_intr5` → `m7b_stopknee_intr5`)

collisions **40 → 28**, completions **4 → 0**, **43 of 48** rollouts braked, median first
hold at control step **88** (p10 16, p90 128, min 16, max 288).

| collision | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 28 | **12** |
| **off: no** | **0** | 8 |

| completion | stop: yes | stop: no |
|---|---|---|
| **off: yes** | 0 | **4** |
| **off: no** | 0 | 44 |

Zero newly-colliding starts in either set: the brake never made a start worse. On the
unshifted set it now prevents 18 of 29 collisions while keeping 12 of 25 completions —
a real trade rather than a blanket halt. Of the 46 braked rollouts, 10 collided anyway:
braking late, or braking in a pose from which the remaining momentum still reaches the
obstacle.

On the shifted scenes it is much weaker: 12 of 40 collisions prevented, and the 4
completions all lost. That is the section 4 transfer gap showing up as behaviour — the
score is a worse proximity estimate on geometry the probe never saw (Spearman +0.480
against +0.868), so the brake fires late and in the wrong places.

**This is a brake, not an avoidance.** It zeroes the pose delta and holds the gripper
command for the rest of the rollout; it has no notion of which direction is safe and it
never resumes. The 13 completions lost on the unshifted set are rollouts that would have
finished and were stopped partway.

### Why not the 90% point (`bench/out/m7b_stop_{ctrl60,intr5}.json`)

The first pass ran `stop` at the 90%-recall threshold with no debounce. Both runs are kept
because the contrast is the argument for section 5's choice:

| | unshifted 90% | unshifted chosen | intrusion-5 90% | intrusion-5 chosen |
|---|---|---|---|---|
| collisions | 29 → **1** | 29 → **11** | 40 → **2** | 40 → **28** |
| completions | 25 → **1** | 25 → **12** | 4 → **0** | 4 → **0** |
| rollouts braked | **60 of 60** | **46 of 60** | **48 of 48** | **43 of 48** |
| median first hold | step **8** | step **92** | step **12** | step **88** |
| newly colliding | 0 | 0 | 0 | 0 |

At 90% recall the monitor braked **every** rollout at a median of control step 8 of 300.
It removed 28 of 29 collisions and 24 of 25 completions by one cause: the arm stopped
moving almost before it started. Reporting "collisions fell 29 → 1" without that would be
a lie of omission. The chosen point brakes 46 of 60 starting at a median step 92, and the
two outcomes finally separate — 18 collisions prevented against 13 completions lost.

For context, the research tree's probe-gated *steering* (not re-run here, and a different
intervention — it steers rather than brakes) reports 14/60 collisions and 18/60
completions against the 29/25 baseline: 0 rescued, 3 lost. The chosen brake reaches 11/60
and 12/60 on the same baseline.

## 10. Reading the artefacts

| file | what |
|---|---|
| `bench/out/m7b_probe.json` | every number in sections 3-5, including both recall sweeps |
| `bench/out/m7b_probe.npz` | the fitted probe and its operating point: `w`, `mu`, `sd`, `ybar`, `alpha`, `threshold`, `debounce` |
| `fastvla/monitor/operating_curve.png` | section 5's figure |
| `bench/out/m7b_intr5_gate.json` | the step-1 gate against the 17500 reference |
| `bench/out/m7b_monitor_cost.json` | section 8's table |
| `bench/out/m7b_stopknee_{ctrl60,intr5}.json` | section 9's matrices at the chosen point, with the per-start keys and the brake-onset distribution |
| `bench/out/m7b_stop_{ctrl60,intr5}.json` | the same at the 90% point, section 9's "why not" |
| `bench/out/m4_accept_M7B_MONITOR_LOG.json` | the parity proof in section 11 |
| `bench/runs/<id>/arrays/<key>.npz` | per rollout: `monitor_scores`, `monitor_alarm`, `monitor_held`, `activations`, `activation_steps`, beside `clearance` and `contact` |

Two fields that sound alike and are not: `monitor_n_held_forwards` counts forwards whose
actions were braked, which is always 0 in `log` mode, and `monitor_n_alarms` counts
debounced alarms. The server now reports the alarm in the response header; before that the
worker could not count alarms at all, never having been told the threshold, and the field
now called `monitor_n_held_forwards` was named `monitor_n_alarms` while holding held
counts. The three runs from the first pass (`m7b_log_ctrl60`, `m7b_stop_ctrl60`,
`m7b_stop_intr5`) carry that old name and have no `monitor_alarm` array; for them, with no
debounce, crossings and alarms are the same thing and
`bench/m7b_monitor_report.py` recomputes both from `monitor_scores`.

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
