# Patches to third-party code

The post-training pipeline needs four edits outside this repository. None of them are
committed upstream, so they are kept here as patches rather than as copied files.

Every patch applies to **openvla-oft at commit `e4287e9`**. Three were produced by
`diff -u` against that tree; two were written by hand in the research repository and are
copied here with their source named. The full narrative account, including what each cost
when it went wrong, is `vla-safety-interp/src/displacement/docs/PATCHES.md`.

| patch | applies to | what it fixes |
|---|---|---|
| `finetune_headinit.patch` | `vla-scripts/finetune.py` → `finetune_headinit.py` | the recipe this pipeline runs |
| `finetune_continue.patch` | `vla-scripts/finetune.py` → `finetune_continue.py` | an earlier variant, kept for the LoRA-init fix |
| `openvla_oft_mixture.patch` | `prismatic/vla/datasets/rlds/dataset.py`, `oxe/configs.py`, `oxe/mixtures.py`, `oxe/transforms.py` | registers the generated dataset and its shared statistics |
| `finetune_final_save.patch` | `vla-scripts/finetune_headinit.py` | saves the last checkpoint before the break |

## `finetune_headinit.patch` — the recipe

Two things, both needed by `run_continued_lora.sh`:

- **Head and proprio warm start.** Adds `--init_heads_from_vla_path` and
  `--init_heads_step`, which load `action_head--<step>_checkpoint.pt` and
  `proprio_projector--<step>_checkpoint.pt` out of the base checkpoint directory instead
  of initialising them randomly. Without this the action head starts from noise while the
  backbone is already trained, and the first thousand steps are spent undoing that.
- **The save condition.** Upstream saves when `log_step % save_freq == 0`. With gradient
  accumulation that is true for *every* micro-batch of the accumulating step, so the same
  checkpoint was written eight times. The patch adds
  `and (batch_idx + 1) % cfg.grad_accumulation_steps == 0`.

The second fix has a sharp edge, and it cost a checkpoint. The `break` at
`log_step == cfg.max_steps` tests only the step count. For any `max_steps` that is a
multiple of `save_freq`, the break can fire on a micro-batch that is **not** an
accumulation boundary, and then the final checkpoint is never written. That is why
`ft_liftonly--3000_chkpt` does not exist while 1000 and 2000 do. Apply
`finetune_final_save.patch` with this one; it is the repair.

## `finetune_final_save.patch` — the repair

Copied from `vla-safety-interp/src/displacement/datagen/finetune_final_save.patch` at
commit `5384bee`. Saves unconditionally at the break, guarded by `if not final_dir.exists()`
so it is a no-op when the periodic save already wrote that step. Written *after* step 3000
was lost; it is not retroactive and recovered nothing.

## `openvla_oft_mixture.patch` — registration and shared statistics

Copied from `vla-safety-interp/src/displacement/datagen/openvla_oft_mixture.patch` at
commit `5384bee`. Four hunks:

- `oxe/configs.py` registers `libero_safety_oa1_liftonly`, including
  `aux_kwargs: {"dataset_statistics": ".../oa1_liftonly_shared_stats.json"}`.
- `oxe/mixtures.py` adds
  `libero_safety_oa1_mix = [("libero_safety", 1.0), ("libero_safety_oa1_liftonly", 67.5)]`.
  The 67.5 is chosen so `balance_weights` lands the generated set at roughly 25% of
  sampled frames, given 99 episodes against the base set's 19,606.
- `oxe/transforms.py` adds the standard LIBERO transform entry for the new key.
- `prismatic/vla/datasets/rlds/dataset.py` adds one line,
  `dataset_kwargs.pop("dataset_statistics", None)`, before the second
  `make_dataset_from_rlds(...)` call in `make_interleaved_dataset`.

That last line is not cosmetic. `make_interleaved_dataset` calls `make_dataset_from_rlds`
twice — once in the statistics loop, once to build the dataset — and the second call
already passes `dataset_statistics=` explicitly. With the override in `aux_kwargs` the
kwarg arrived twice and raised `TypeError: got multiple values for keyword argument
'dataset_statistics'`. **This crashed a launched run**, because the verification had
exercised only the first call site.

## `finetune_continue.patch` — kept for one line

An earlier continued-LoRA variant, not the recipe used here. It is kept because of its
LoRA initialisation: upstream uses `init_lora_weights="gaussian"`, which means the adapter
is *not* the identity at step 0 and silently perturbs a checkpoint you meant to continue
from. The patch sets `init_lora_weights=True` so `lora_B` starts at zero and the adapter
is a no-op until it learns something. It also adds a validation gate — 64 forward passes
before any training, aborting if the mean loss exceeds a threshold — which is how you find
out that your "resume" is not faithful *before* spending four hours.

## Applying them

```bash
cd $POSTTRAIN_OPENVLA_OFT                     # openvla-oft at e4287e9
patch -p0 < .../patches/openvla_oft_mixture.patch
cp vla-scripts/finetune.py vla-scripts/finetune_headinit.py
patch -p0 vla-scripts/finetune_headinit.py < .../patches/finetune_headinit.patch
patch -p0 vla-scripts/finetune_headinit.py < .../patches/finetune_final_save.patch
```

`finetune_headinit.patch` is a diff from `finetune.py`, so it is applied to a copy of that
file, not to the original.
