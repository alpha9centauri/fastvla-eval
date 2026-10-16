# Post-training: teaching the policy to lift over the obstacle

Part 3 of this repository. The pipeline that took the base policy, built a dataset of
planner-generated demonstrations, and continued its LoRA fine-tune on a mixture of the two.
The scripts are in `fastvla/posttrain/`; every one names the file and commit it was copied
from. No training runs from this repository — it is the recorded recipe, and one
evaluation of its output.

Every number below is from a log or an artefact, named inline.

## 1. Data conversion

The benchmark ships as LeRobot parquet plus MP4. OpenVLA-OFT trains from RLDS/TFDS. The
converter (`convert_lerobot_to_rlds.py`) decodes the videos with PyAV, pairs frames with
actions, and writes a real TFDS build — not a pile of TFRecords — so the training loader
can open it.

| | |
|---|---|
| trajectories | **19,606** (17,645 train + 1,961 validation) |
| transitions | **3,431,183** |
| on disk | 278.7 GB (238.1 train + 40.6 validation) |
| source episodes dropped | 58 of 19,664, verified black frames; PyAV and ffmpeg agree (`full_build.log`) |

Counts read from `rlds_full/libero_safety/1.0.0/dataset_info.json` and the
`dataset_statistics_*.json` beside it. The dropped-episode note is from
`vla-safety-interp/notes/libero-oft-training-constraints.md`.

### The parallel build

`build_rlds_parallel.py` gives each worker a disjoint slice of episodes and has it
serialise straight into its own shards, using the builder's own feature spec. One final
pass records shard lengths and byte counts through TFDS's own `DatasetInfo` writer, which
is what makes the result a legitimate TFDS build.

Measured on the same 500 episodes, 8 workers against one process:

| build | wall clock | source |
|---|---|---|
| single process | **37.96 min** | `rlds500.log`, `Generating splits...: 2/2 [37:57<00:00]` |
| 8 workers | **7.0 min** | `p500.log`, `build complete in 7.0 min` |

**5.4x.** The full 19,664-episode build took **259.8 min** with 8 workers
(`full_build.log`), and the same log records the 58 skipped episodes and the resulting
17,645 + 1,961 = 19,606. At the single-process rate the full build would have been about
**25 hours** — an extrapolation from the 500-episode run, not a measurement; nobody ran
the full set single-process.

It writes **the same bytes**, redistributed. Verified here by comparing a single-process
build against a parallel build of the same input:

| pair | shards | total TFRecord bytes | trajectories | transitions |
|---|---|---|---|---|
| `rlds_500` (single) | 72 | **7,101,305,880** | 500 | 87,064 |
| `rlds_500p` (parallel) | 16 | **7,101,305,880** | 500 | 87,064 |
| `rlds_smoke` (single) | 3 | **300,176,872** | 20 | 3,446 |
| `rlds_ptest` (parallel) | 6 | **300,176,872** | 20 | 3,446 |

Equal to the byte in total payload, with a different shard count — so the individual files
are *not* byte-identical and cannot be, because the parallel builder writes one shard set
per worker. The serialised content is the same; the partitioning is not.

`dataset_statistics` is deliberately **not** written by the parallel builder. OpenVLA's
loader computes it over the whole dataset on first load and caches it under a hash of the
builder info; precomputing it risks disagreeing with what the loader would derive.

## 2. Registration, and the statistic that has to be shared

A new dataset reaches the training mixture through three registrations in openvla-oft —
`oxe/configs.py`, `oxe/mixtures.py`, `oxe/transforms.py` — plus one fix in the interleaving
code. All four are `fastvla/posttrain/patches/openvla_oft_mixture.patch`; what each does and
why is `fastvla/posttrain/patches/README.md`.

The mixture is `libero_safety_oa1_mix = [("libero_safety", 1.0),
("libero_safety_oa1_liftonly", 67.5)]`. The 67.5 is chosen so that `balance_weights`, which
multiplies weights by transition counts and renormalises, lands the generated set at roughly
**25% of sampled frames** given 99 episodes against the base set's 19,606.

The generated set is **99 trajectories, 16,954 transitions** (80 train + 19 validation,
`data/ft_liftonly_data/libero_safety_oa1_liftonly/1.0.0/`).

**The statistics have to be the base set's, not the new set's.** Actions are normalised by
per-dimension q01/q99 before they become tokens. The two sets disagree sharply:

| | action q01, first three dims | action q99, first three dims |
|---|---|---|
| base, 3,431,183 transitions | −0.0138, −0.0213, −0.0156 | 0.0136, 0.0221, 0.0198 |
| generated set's own, 16,954 | −0.0044, −0.0075, −0.0100 | 0.0076, 0.0070, 0.0100 |

The generated demos are smooth planner trajectories, so their action range is about half the
base set's. Normalised by their own statistics, the same physical motion would encode to a
different token, and the mixture would be teaching two different action languages at once.
`oa1_liftonly_shared_stats.json` therefore carries the **base set's q01/q99** with the
generated set's transition count. Verified: that file's q01/q99 equal the base set's exactly
and differ from the generated set's own.

## 3. Fitting a 7B LoRA fine-tune on one 24 GB card

The model is 7.65 B parameters. What is trained is a rank-32 LoRA adapter plus two heads:

| part | trainable parameters |
|---|---|
| LoRA adapter (r32, α16, dropout 0) | 110,828,288 |
| action head | 151,117,831 |
| proprio projector | 16,818,176 |
| **total** | **278,764,295** (3.6% of 7,652,065,472) |

From the run log, `vla-safety-interp/src/displacement/ft_liftonly.log`.

Per-step batch size is **1** — memory-capped, not chosen — and an effective batch of 8 comes
from 8 gradient-accumulation micro-steps. Everything else follows from that constraint.

**Peak GPU memory was not recorded for this run**, and is not stated here. What the logs do
record is the shape of the constraint: batch size is capped at 1 by memory, and writing a
checkpoint costs ~24 minutes and 51.3 GB of peak host RSS
(`notes/libero-oft-training-constraints.md`).

## 4. The base checkpoint, and why 17500

The base policy came from an earlier fine-tune evaluated at three checkpoints, 80 rollouts
each (`vla-safety-interp/notes/libero-oft-headinit-run.md`):

| step | success | collisions | mean clearance | obstacle avoidance |
|---|---|---|---|---|
| 12500 | 25.0% ± 4.8% | 58 | 0.74 cm | 0/60 |
| 15000 | 36.3% ± 5.4% | 32 | 3.64 cm | 12/60 |
| **17500** | **56.3% ± 5.5%** | 36 | 2.38 cm | **26/60** |

17500 was chosen as the best of the ladder and the last complete checkpoint of that run.
The note alongside it is worth repeating: at 12500 the policy looked permanently bimodal —
20/20 on human safety, 0/60 on obstacle avoidance — and that was recorded as a capability
ceiling. It was not. Between 12500 and 17500 one task went 0/10 to 9/10. The checkpoint was
mid-transition, not plateaued.

## 5. The continued-LoRA post-training

`fastvla/posttrain/run_continued_lora.sh`. Continued LoRA, not a resumed optimiser: the
17500 run directory is loaded as the base — its adapter is already merged into those
safetensors — the action head and proprio projector are warm-started from its step-17500
checkpoints, and a **fresh zero-init adapter** is attached on top. `lora_B` starts at zero,
so the adapter is a no-op at step 0 and the run begins exactly at the base policy.

| | |
|---|---|
| data | `libero_safety_oa1_mix`, 25% generated / 75% original |
| adapter | rank 32, α 16, dropout 0 |
| learning rate | 5e-5, decay after 2500 |
| batch | 1 × 8 accumulation = effective 8 |
| steps | 3000, saving every 1000 |
| wall clock | **4 h 25 m 30 s**, **4.83 s/it** |
| throughput | **1.66 samples/s** at effective batch 8 (8 ÷ 4.83) |
| checkpoint | **15 GB** each |

Loss, as the mean of `curr_action_l1_loss` over a ±50-step window, recomputed here from
`ft_liftonly.log` and matching `RESULTS.md` §11 to within 3e-4:

| step | 0 | 1000 | 2000 | 3000 |
|---|---|---|---|---|
| L1 loss | 0.0495 | 0.0358 | 0.0209 | 0.0224 |

A 58% fall to step 2000, flat and slightly worse after. **Step 2000 is the checkpoint
evaluated below**, and the loss curve is the reason: 3000 was not an improvement.

There is no step-3000 checkpoint at all. The save condition and the `break` disagreed about
gradient-accumulation boundaries, so the final save never fired. The defect, the fix, and
what it cost are in `fastvla/posttrain/patches/README.md`.

## 6. What the post-trained checkpoint actually does

`ft_liftonly--2000` through this harness, 60 starts, 8 workers, batch 1, beside the base
17500 from the same harness (`bench/out/p3_ft2000_vs_17500.json`).

| | base 17500 | post-trained ft2000 |
|---|---|---|
| collisions | 29 | **21** |
| completions | 25 | **20** |
| mean minimum clearance | 2.35 cm | 2.48 cm |

Per start, which is the part that matters:

| collision | ft: yes | ft: no |
|---|---|---|
| **base: yes** | 17 | **12 prevented** |
| **base: no** | **4 created** | 27 |

| completion | ft: yes | ft: no |
|---|---|---|
| **base: yes** | 17 | **8 lost** |
| **base: no** | **3 gained** | 32 |

Eight fewer collisions is a net of 12 prevented against 4 created, and it costs five
completions net. On the 12 starts where contact is prevented the arm does not halt — those
rollouts run *longer* than the base's (295.5 against 288.6 control steps) and end at 1.48 cm
of clearance instead of touching, which is avoidance rather than a stop. The eight lost
completions all run the full 300-step cap with the arm still moving, so they are time-outs,
not early stops: across all 60 starts the post-trained policy travels **more** in the last
100 steps than the base (52.8 cm against 41.8 cm). The lift the generated demos taught shows
up where it was least needed — maximum end-effector height rises 22 mm on the 27 starts
where both policies already clear the obstacle, against 3.4 mm averaged over all 60 and
−33 mm on the 12 starts where contact was actually prevented. The post-training moved the
policy, and the direction is net positive on collisions, but the mechanism is not the one
the demonstrations were designed to teach.

## 7. Reading the artefacts

| file | what |
|---|---|
| `bench/out/p3_ft2000_vs_17500.json` | section 6, with the per-start keys of every cell |
| `bench/runs/p3_ft2000/` | the evaluation run: merged rows and per-rollout arrays |
| `audit/P3_FT2000` | its run log |
| `fastvla/posttrain/config.env` | every path the pipeline needs, with this box's values as the example |
| `fastvla/posttrain/patches/README.md` | the four third-party patches and what each fixes |

### Where the copied scripts came from

| file here | copied from | commit |
|---|---|---|
| `convert_lerobot_to_rlds.py` | `/data_new/vchandr2/libero_safety_converter_tfds.py` | not version-controlled; md5 `db0e65d28f2b79e36dadb909c1a3c743` |
| `build_rlds_parallel.py` | `vla-safety-interp/src/build_rlds_parallel.py` | `e602fd1` |
| `run_continued_lora.sh` | `vla-safety-interp/src/displacement/datagen/run_ft_liftonly.sh` | `5384bee` |
| `merge_lora.py` | `openvla-oft/vla-scripts/merge_lora_weights_and_save.py` | `e4287e9`, unmodified upstream (MIT) |
| `patches/openvla_oft_mixture.patch` | `vla-safety-interp/src/displacement/datagen/` | `5384bee` |
| `patches/finetune_final_save.patch` | same | `5384bee` |
| `patches/finetune_headinit.patch` | generated by `diff -u` against `openvla-oft` | `e4287e9` |
| `patches/finetune_continue.patch` | same | `e4287e9` |

The only edits to copied code are the paths: every hardcoded `/data_new/...` now comes from
`config.env`. Those three files are excluded from this repository's formatter, so they still
diff cleanly against their originals.
