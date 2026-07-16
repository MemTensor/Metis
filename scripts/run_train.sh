#!/bin/bash
# Metis unified multi-task training runner.
#
# Default architecture: NormedReweightLearnedQueryMetisBlock + AlphaTopP hyper-memory
#                       + NormalizedDeltaNetMetisLocalMemory
#
# Usage:
#   bash scripts/run_train.sh                          # default config
#   NPROC_PER_NODE=4 bash scripts/run_train.sh         # override GPU count
#   GRAD_ACCUM=30 bash scripts/run_train.sh            # override grad accum
#   USE_WANDB=0 bash scripts/run_train.sh              # disable wandb

set -euo pipefail
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Distributed config ──────────────────────────────────────────────
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29540}

# ── Training hyperparams ────────────────────────────────────────────
GRAD_ACCUM=${GRAD_ACCUM:-10}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-2e-4}
NUM_EPOCHS=${NUM_EPOCHS:-2}
MAX_STEPS=${MAX_STEPS:-0}
WARMUP_STEPS=${WARMUP_STEPS:-200}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-1024}
SAVE_STEPS=${SAVE_STEPS:-2000}

# Checkpoint format: "delta" = only trainable Metis weights (default), ~100x
# smaller since the frozen backbone is not duplicated per checkpoint;
# "full" = legacy save_pretrained dump.
CHECKPOINT_SAVE_MODE=${CHECKPOINT_SAVE_MODE:-delta}

# ── Model architecture ──────────────────────────────────────────────
METIS_BLOCK_TYPE=${METIS_BLOCK_TYPE:-NormedReweightLearnedQueryMetisBlock}
METIS_HYPER_MEMORY_TYPE=${METIS_HYPER_MEMORY_TYPE:-StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory}
METIS_LOCAL_MEMORY_TYPE=${METIS_LOCAL_MEMORY_TYPE:-NormalizedDeltaNetMetisLocalMemory}
STRIDE_INTERVAL=${STRIDE_INTERVAL:-8}
POOL_TEMPERATURE=${POOL_TEMPERATURE:-1.0}
METIS_REWEIGHT_GAMMA=${METIS_REWEIGHT_GAMMA:-0.9}
UPDATE_RATIO=${UPDATE_RATIO:-0.9}
COMMIT_HIDDEN_OFFSET=${COMMIT_HIDDEN_OFFSET:-0}
MEM_NORM_INIT=${MEM_NORM_INIT:-1.0}
UNIFORM_NUM_SELECTED=${UNIFORM_NUM_SELECTED:-16}
ALPHA_TOP_P=${ALPHA_TOP_P:-0.9}
ALPHA_MIN_TOKENS=${ALPHA_MIN_TOKENS:-1}
ALPHA_MAX_TOKENS=${ALPHA_MAX_TOKENS:-0}
ALPHA_MAX_FRACTION=${ALPHA_MAX_FRACTION:-0.0}
QK_KERNEL_TYPE=${QK_KERNEL_TYPE:-elu_plus_one}
GATED_DELTA_ALPHA_INIT=${GATED_DELTA_ALPHA_INIT:-1.0}
GATED_DELTA_BETA_INIT=${GATED_DELTA_BETA_INIT:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ── Task schedule ───────────────────────────────────────────────────
TASKS=${TASKS:-0,1,2,3,4}
TASK0_WEIGHT_START=${TASK0_WEIGHT_START:-0.25}
TASK0_WEIGHT_END=${TASK0_WEIGHT_END:-0.1}
TASK1_WEIGHT_START=${TASK1_WEIGHT_START:-0.35}
TASK1_WEIGHT_END=${TASK1_WEIGHT_END:-0.25}
TASK2_WEIGHT_START=${TASK2_WEIGHT_START:-0.2}
TASK2_WEIGHT_END=${TASK2_WEIGHT_END:-0.3}
TASK3_WEIGHT_START=${TASK3_WEIGHT_START:-0.1}
TASK3_WEIGHT_END=${TASK3_WEIGHT_END:-0.2}
TASK4_WEIGHT_START=${TASK4_WEIGHT_START:-0.1}
TASK4_WEIGHT_END=${TASK4_WEIGHT_END:-0.15}

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-}
BACKBONE_TYPE=${BACKBONE_TYPE:-qwen3_5}

# ── Resume / warm-start ─────────────────────────────────────────────
# RESUME_FROM_CHECKPOINT: dir, or "auto" to pick the newest resumable
#   checkpoint-N under ${OUTPUT_DIR}/${NAME} (full state: optimizer/RNG/step).
# INIT_FROM_CHECKPOINT: weights-only warm start, fresh optimizer/step.
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
INIT_FROM_CHECKPOINT=${INIT_FROM_CHECKPOINT:-}
# Required for resume: lets HF Trainer torch.load optimizer/rng files that
# sit inside the (trusted, local) checkpoint dir. No effect otherwise.
export METIS_TRUST_LOCAL_TORCH_LOAD=${METIS_TRUST_LOCAL_TORCH_LOAD:-1}
DATA_DIR=${DATA_DIR-}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR-}
EVAL_DATA_DIR=${EVAL_DATA_DIR-}
EVAL_TOKENIZED_DATA_DIR=${EVAL_TOKENIZED_DATA_DIR-}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoint}
DS_CONFIG=${DS_CONFIG:-}
NAME=${NAME:-}

if [ -z "${MODEL_PATH}" ] || [ -z "${NAME}" ]; then
    echo "MODEL_PATH and NAME must be set." >&2
    exit 2
fi
if [ -z "${DATA_DIR}" ] && [ -z "${TOKENIZED_DATA_DIR}" ]; then
    echo "DATA_DIR or TOKENIZED_DATA_DIR must be set." >&2
    exit 2
fi
if [ -z "${EVAL_DATA_DIR}" ] && [ -z "${EVAL_TOKENIZED_DATA_DIR}" ]; then
    echo "EVAL_DATA_DIR or EVAL_TOKENIZED_DATA_DIR must be set." >&2
    exit 2
fi

# ── Wandb ───────────────────────────────────────────────────────────
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-metis_training}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}}
EVAL_STEPS=${EVAL_STEPS:-1000}
GEN_EVAL_STEPS=${GEN_EVAL_STEPS:-0}
EVAL_SAMPLES=${EVAL_SAMPLES:-60}
EVAL_SAMPLES_PER_TASK=${EVAL_SAMPLES_PER_TASK:-20}
EVAL_MAX_SAMPLES_PER_TASK=${EVAL_MAX_SAMPLES_PER_TASK:-0}
EVAL_MAX_TOTAL_TOKENS=${EVAL_MAX_TOTAL_TOKENS:--1}
MAX_NEW_TOKENS_EVAL=${MAX_NEW_TOKENS_EVAL:-512}

# ── Launch ──────────────────────────────────────────────────────────
TRAIN_ARGS=(
    --model_path "${MODEL_PATH}"
    --backbone_type "${BACKBONE_TYPE}"
    --metis_block_type "${METIS_BLOCK_TYPE}"
    --metis_hyper_memory_type "${METIS_HYPER_MEMORY_TYPE}"
    --metis_local_memory_type "${METIS_LOCAL_MEMORY_TYPE}"
    --update_ratio "${UPDATE_RATIO}"
    --commit_hidden_offset "${COMMIT_HIDDEN_OFFSET}"
    --mem_norm_init "${MEM_NORM_INIT}"
    --uniform_num_selected "${UNIFORM_NUM_SELECTED}"
    --stride_interval "${STRIDE_INTERVAL}"
    --pool_temperature "${POOL_TEMPERATURE}"
    --alpha_top_p "${ALPHA_TOP_P}"
    --alpha_min_tokens "${ALPHA_MIN_TOKENS}"
    --alpha_max_tokens "${ALPHA_MAX_TOKENS}"
    --alpha_max_fraction "${ALPHA_MAX_FRACTION}"
    --qk_kernel_type "${QK_KERNEL_TYPE}"
    --gated_delta_alpha_init "${GATED_DELTA_ALPHA_INIT}"
    --gated_delta_beta_init "${GATED_DELTA_BETA_INIT}"
    --metis_reweight_gamma "${METIS_REWEIGHT_GAMMA}"
    --data_dir "${DATA_DIR}"
    --eval_data_dir "${EVAL_DATA_DIR}"
    --max_total_tokens "${MAX_TOTAL_TOKENS}"
    --eval_max_samples_per_task "${EVAL_MAX_SAMPLES_PER_TASK}"
    --eval_max_total_tokens "${EVAL_MAX_TOTAL_TOKENS}"
    --output_dir "${OUTPUT_DIR}/${NAME}"
    --num_epochs "${NUM_EPOCHS}"
    --max_steps "${MAX_STEPS}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --learning_rate "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --warmup_steps "${WARMUP_STEPS}"
    --tasks "${TASKS}"
    --task0_weight_start "${TASK0_WEIGHT_START}"
    --task0_weight_end "${TASK0_WEIGHT_END}"
    --task1_weight_start "${TASK1_WEIGHT_START}"
    --task1_weight_end "${TASK1_WEIGHT_END}"
    --task2_weight_start "${TASK2_WEIGHT_START}"
    --task2_weight_end "${TASK2_WEIGHT_END}"
    --task3_weight_start "${TASK3_WEIGHT_START}"
    --task3_weight_end "${TASK3_WEIGHT_END}"
    --task4_weight_start "${TASK4_WEIGHT_START}"
    --task4_weight_end "${TASK4_WEIGHT_END}"
    --lora_r 0
    --log_steps 5
    --save_steps "${SAVE_STEPS}"
    --checkpoint_save_mode "${CHECKPOINT_SAVE_MODE}"
    --eval_steps "${EVAL_STEPS}"
    --gen_eval_steps "${GEN_EVAL_STEPS}"
    --eval_samples "${EVAL_SAMPLES}"
    --eval_samples_per_task "${EVAL_SAMPLES_PER_TASK}"
    --max_new_tokens_eval "${MAX_NEW_TOKENS_EVAL}"
    --dtype bfloat16
    --seed 42
)

[ -n "${RESUME_FROM_CHECKPOINT}" ] && TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
[ -n "${INIT_FROM_CHECKPOINT}" ] && TRAIN_ARGS+=(--init_from_checkpoint "${INIT_FROM_CHECKPOINT}")
[ -n "${TOKENIZED_DATA_DIR}" ] && TRAIN_ARGS+=(--tokenized_data_dir "${TOKENIZED_DATA_DIR}")
[ -n "${EVAL_TOKENIZED_DATA_DIR}" ] && TRAIN_ARGS+=(--eval_tokenized_data_dir "${EVAL_TOKENIZED_DATA_DIR}")
[ -n "${DS_CONFIG}" ] && TRAIN_ARGS+=(--deepspeed "${DS_CONFIG}")
if [ "${USE_WANDB}" = "1" ]; then
    TRAIN_ARGS+=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
fi

exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    train/run_train.py \
    "${TRAIN_ARGS[@]}" \
    "$@"
