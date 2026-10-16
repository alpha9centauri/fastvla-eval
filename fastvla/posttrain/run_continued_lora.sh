#!/usr/bin/env bash
# Copied from vla-safety-interp/src/displacement/datagen/run_ft_liftonly.sh at commit
# 5384bee (that tree is read-only for this project). The recipe and every flag are
# unchanged; the box's paths now come from config.env instead of being written in here.
#
# ft_liftonly: continued LoRA from the 17500 checkpoint on a 25/75 mix of the
# planner-generated oa1 lift-only set and the original LIBERO-Safety demos.
#
# Recipe: rank 32, alpha 16, dropout 0, lr 5e-5, batch 1 x accum 8, lora_B zero-init so the
# adapter is a no-op at step 0, BC loss only. Everything else follows the run that produced
# 17500.
#
# "Resume from 17500" means: load the 17500 run directory as the base -- its LoRA is already
# merged into those safetensors because finetune.py merges during training -- initialise the
# action head and proprio projector from its step-17500 checkpoints, and attach a FRESH
# zero-init adapter on top. That is continued LoRA, not a resumed optimiser state.
#
# Needs vla-scripts/finetune_headinit.py in the openvla-oft tree: that is upstream
# finetune.py with patches/finetune_headinit.patch applied. See patches/README.md.
set -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
set -a; . "${POSTTRAIN_CONFIG:-$HERE/config.env}"; set +a

cd "$POSTTRAIN_OPENVLA_OFT" || exit 1

export CUDA_VISIBLE_DEVICES="$POSTTRAIN_CUDA_DEVICE"
export TORCH_CUDNN_V8_API_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "$POSTTRAIN_RUN_ROOT"

"$POSTTRAIN_PYTHON" \
  --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_headinit.py \
  --vla_path "$POSTTRAIN_BASE_CKPT" \
  --data_root_dir "$POSTTRAIN_MIX_ROOT" \
  --dataset_name libero_safety_oa1_mix \
  --run_root_dir "$POSTTRAIN_RUN_ROOT" \
  --init_heads_from_vla_path True --init_heads_step "$POSTTRAIN_BASE_STEP" \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 2 --use_proprio True \
  --batch_size 1 --grad_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --lora_rank 32 --lora_alpha 16 --lora_dropout 0.0 \
  --num_steps_before_decay 2500 --max_steps 3000 \
  --save_freq 1000 --save_latest_checkpoint_only False \
  --image_aug True \
  --run_id_note ft_liftonly
rc=$?
[ $rc -ne 0 ] && { echo "[ALERT] ft_liftonly FAILED|exited $rc"; exit $rc; }
# List the checkpoints that are ON DISK. An earlier version asserted "1000/2000/3000"
# unconditionally and was wrong: the break in finetune_headinit.py could fire before the
# final save, so step 3000 was never written and the alert said it had been.
FOUND=$(ls -d "$POSTTRAIN_RUN_ROOT"/*ft_liftonly--*_chkpt 2>/dev/null \
        | sed -E 's/.*ft_liftonly--([0-9]+)_chkpt/\1/' | sort -n | tr '\n' ' ')
[ -z "$FOUND" ] && FOUND="NONE"
echo "[ALERT] ft_liftonly done|checkpoints on disk: $FOUND"
