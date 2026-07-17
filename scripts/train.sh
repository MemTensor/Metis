#!/bin/bash
# Metis training launcher.
#
# Configuration reference
# =======================
#
# Required launcher arguments (no defaults):
#   --model-path PATH       Local backbone path or Hugging Face model ID.
#   --name NAME             Training-run identifier.
#   --train-data PATH       Tokenized cache, or raw JSONL with --data-format raw.
#   --valid-data PATH       Tokenized validation cache or raw validation data.
#
# Launcher options and defaults:
#   --data-format            tokenized (alternative: raw)
#   --backbone-type          qwen3_5 (alternatives: qwen3, llama)
#   --output-dir             checkpoints
#   --cuda-visible-devices   0,1,2,3
#   --nproc-per-node         4
#   --batch-size             2 per device
#   --grad-accum             10; effective batch = workers * batch * accumulation
#   --deepspeed              disabled unless a config path is supplied
#   --resume-from-checkpoint unset; "auto" selects the newest checkpoint
#   --init-from-checkpoint   unset; loads weights without optimizer/run state
#
# Environment-variable defaults:
#   MASTER_PORT=29540                  MAX_TOTAL_TOKENS=1024
#   TASKS=0,1,2,3,4                   LR=2e-4
#   WEIGHT_DECAY=0.01                 MAX_GRAD_NORM=1.0
#   WARMUP_STEPS=200                  NUM_EPOCHS=2
#   MAX_STEPS=0                       SAVE_STEPS=2000
#   CHECKPOINT_SAVE_MODE=delta        EVAL_STEPS=1000
#   GEN_EVAL_STEPS=0                  EVAL_SAMPLES=60
#   EVAL_SAMPLES_PER_TASK=20          USE_WANDB=1
#   WANDB_PROJECT=metis_training
#
# Memory-architecture defaults:
#   METIS_BLOCK_TYPE=NormedReweightLearnedQueryMetisBlock
#   METIS_HYPER_MEMORY_TYPE=StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory
#   METIS_LOCAL_MEMORY_TYPE=NormalizedDeltaNetMetisLocalMemory
#   UPDATE_RATIO=0.9                  COMMIT_HIDDEN_OFFSET=0
#   STRIDE_INTERVAL=8                 POOL_TEMPERATURE=1.0
#   METIS_REWEIGHT_GAMMA=0.9          ALPHA_TOP_P=0.9
#   QK_KERNEL_TYPE=elu_plus_one       GATED_DELTA_ALPHA_INIT=1.0
#   GATED_DELTA_BETA_INIT=1.0
#
# Common hyper-memory variants:
#   LastTokenGatedDeltaRuleMetisHyperMemory
#   StraightThroughAlphaTopPKeyNormMetisHyperMemory
#
# Task schedule (START -> END over training):
#   Task 0  0.25 -> 0.10   reconstruction and explicit/implicit fact recall
#   Task 1  0.35 -> 0.25   remember, forget, update, and reflection operations
#   Task 2  0.20 -> 0.30   distractor and long-context memory operations
#   Task 3  0.10 -> 0.20   mixed and LLM-snippet memory interactions
#   Task 4  0.10 -> 0.15   normal/no-query examples for memory regularization
#
# The backbone is frozen and LoRA is disabled. Only native-memory parameters
# are optimized. run_train.py additionally defaults to seed=42, a
# constant-with-warmup scheduler, and lora_r=0.
#
# Set environment variables before the command to override recipe settings.
# Explicit launcher options take precedence over matching environment values.
# Pass arguments after -- directly to train/run_train.py. Set DRY_RUN=1 to
# print the resolved configuration without launching training.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train.sh \
    --model-path PATH \
    --name NAME \
    --train-data PATH \
    --valid-data PATH \
    [options]

Required:
  --model-path PATH             Backbone model path or Hugging Face model ID.
  --name NAME                   Run name and checkpoint subdirectory.
  --train-data PATH             Training data directory.
  --valid-data PATH             Validation data directory.

Options:
  --data-format FORMAT          tokenized or raw (default: tokenized).
  --backbone-type TYPE          qwen3_5, qwen3, or llama (default: qwen3_5).
  --output-dir PATH             Checkpoint root directory.
  --cuda-visible-devices LIST   CUDA device list (default: 0,1,2,3).
  --nproc-per-node N            Number of torchrun workers (default: 4).
  --batch-size N                Per-device batch size (default: 2).
  --grad-accum N                Gradient accumulation steps (default: 10).
  --metis-block-type TYPE       Metis block implementation.
  --metis-hyper-memory-type TYPE
                                Hyper-memory implementation.
  --metis-local-memory-type TYPE
                                Local-memory implementation.
  --deepspeed PATH              Enable DeepSpeed with this config.
  --no-deepspeed                Disable DeepSpeed explicitly.
  --resume-from-checkpoint PATH Use PATH or auto for a full-state resume.
  --init-from-checkpoint PATH   Load weights only and start a new run state.
  -h, --help                    Show this message.

Values passed on the command line override matching environment variables.
Use -- before extra train/run_train.py arguments.
EOF
}

require_value() {
  if [ "$#" -lt 2 ] || [ -z "${2}" ] || [[ "${2}" == --* ]]; then
    echo "Error: ${1} requires a value." >&2
    usage >&2
    exit 2
  fi
}

MODEL_PATH=${MODEL_PATH:-}
NAME=${NAME:-${RUN_NAME:-}}
TRAIN_DATA=${TRAIN_DATA:-${TOKENIZED_DATA_DIR:-}}
VALID_DATA=${VALID_DATA:-${EVAL_TOKENIZED_DATA_DIR:-}}
DATA_FORMAT=${DATA_FORMAT:-tokenized}
BACKBONE_TYPE=${BACKBONE_TYPE:-qwen3_5}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints}
DS_CONFIG=${DS_CONFIG:-}
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path)
      require_value "$@"; MODEL_PATH=$2; shift 2 ;;
    --name)
      require_value "$@"; NAME=$2; shift 2 ;;
    --train-data)
      require_value "$@"; TRAIN_DATA=$2; shift 2 ;;
    --valid-data)
      require_value "$@"; VALID_DATA=$2; shift 2 ;;
    --data-format)
      require_value "$@"; DATA_FORMAT=$2; shift 2 ;;
    --backbone-type)
      require_value "$@"; BACKBONE_TYPE=$2; shift 2 ;;
    --output-dir)
      require_value "$@"; OUTPUT_DIR=$2; shift 2 ;;
    --cuda-visible-devices)
      require_value "$@"; CUDA_VISIBLE_DEVICES=$2; shift 2 ;;
    --nproc-per-node)
      require_value "$@"; NPROC_PER_NODE=$2; shift 2 ;;
    --batch-size)
      require_value "$@"; BATCH_SIZE=$2; shift 2 ;;
    --grad-accum)
      require_value "$@"; GRAD_ACCUM=$2; shift 2 ;;
    --metis-block-type)
      require_value "$@"; METIS_BLOCK_TYPE=$2; shift 2 ;;
    --metis-hyper-memory-type)
      require_value "$@"; METIS_HYPER_MEMORY_TYPE=$2; shift 2 ;;
    --metis-local-memory-type)
      require_value "$@"; METIS_LOCAL_MEMORY_TYPE=$2; shift 2 ;;
    --deepspeed)
      require_value "$@"; DS_CONFIG=$2; shift 2 ;;
    --no-deepspeed)
      DS_CONFIG=; shift ;;
    --resume-from-checkpoint)
      require_value "$@"; RESUME_FROM_CHECKPOINT=$2; shift 2 ;;
    --init-from-checkpoint)
      require_value "$@"; INIT_FROM_CHECKPOINT=$2; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; EXTRA_ARGS=("$@"); break ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

missing=()
[ -n "${MODEL_PATH}" ] || missing+=(--model-path)
[ -n "${NAME}" ] || missing+=(--name)
[ -n "${TRAIN_DATA}" ] || missing+=(--train-data)
[ -n "${VALID_DATA}" ] || missing+=(--valid-data)
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Error: missing required arguments: ${missing[*]}" >&2
  usage >&2
  exit 2
fi

case "${DATA_FORMAT}" in
  tokenized|raw) ;;
  *) echo "Error: --data-format must be tokenized or raw." >&2; exit 2 ;;
esac

case "${BACKBONE_TYPE}" in
  qwen3_5|qwen3|llama) ;;
  *) echo "Error: --backbone-type must be qwen3_5, qwen3, or llama." >&2; exit 2 ;;
esac

for value_name in NPROC_PER_NODE BATCH_SIZE GRAD_ACCUM; do
  value=${!value_name:-}
  if [ -n "${value}" ] && ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${value_name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

# ── Environment ─────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── Distributed ─────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export MASTER_PORT=${MASTER_PORT:-29540}

# ── Model and data ──────────────────────────────────────────────────
export MODEL_PATH BACKBONE_TYPE DS_CONFIG
if [ "${DATA_FORMAT}" = "tokenized" ]; then
  export TOKENIZED_DATA_DIR=${TRAIN_DATA}
  export EVAL_TOKENIZED_DATA_DIR=${VALID_DATA}
  export DATA_DIR=${DATA_DIR:-${TRAIN_DATA}}
  export EVAL_DATA_DIR=${EVAL_DATA_DIR:-${VALID_DATA}}
else
  export DATA_DIR=${TRAIN_DATA}
  export EVAL_DATA_DIR=${VALID_DATA}
  export TOKENIZED_DATA_DIR=
  export EVAL_TOKENIZED_DATA_DIR=
fi
export TASKS=${TASKS:-0,1,2,3,4}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-1024}

# ── Metis memory architecture (GDN recipe) ──────────────────────────
export METIS_BLOCK_TYPE=${METIS_BLOCK_TYPE:-NormedReweightLearnedQueryMetisBlock}
export METIS_HYPER_MEMORY_TYPE=${METIS_HYPER_MEMORY_TYPE:-StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory}
export METIS_LOCAL_MEMORY_TYPE=${METIS_LOCAL_MEMORY_TYPE:-NormalizedDeltaNetMetisLocalMemory}
export UPDATE_RATIO=${UPDATE_RATIO:-0.9}
export COMMIT_HIDDEN_OFFSET=${COMMIT_HIDDEN_OFFSET:-0}
export STRIDE_INTERVAL=${STRIDE_INTERVAL:-8}
export POOL_TEMPERATURE=${POOL_TEMPERATURE:-1.0}
export METIS_REWEIGHT_GAMMA=${METIS_REWEIGHT_GAMMA:-0.9}
export ALPHA_TOP_P=${ALPHA_TOP_P:-0.9}
export QK_KERNEL_TYPE=${QK_KERNEL_TYPE:-elu_plus_one}
export GATED_DELTA_ALPHA_INIT=${GATED_DELTA_ALPHA_INIT:-1.0}
export GATED_DELTA_BETA_INIT=${GATED_DELTA_BETA_INIT:-1.0}

# ── Optimization ────────────────────────────────────────────────────
# Effective batch is NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM.
# MasterWeightAdamW is selected automatically for bf16 without DeepSpeed.
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-10}
export LR=${LR:-2e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
export WARMUP_STEPS=${WARMUP_STEPS:-200}
export NUM_EPOCHS=${NUM_EPOCHS:-2}
export MAX_STEPS=${MAX_STEPS:-0}

# ── Task weight schedule (recipe §3.3) ──────────────────────────────
export TASK0_WEIGHT_START=${TASK0_WEIGHT_START:-0.25}
export TASK0_WEIGHT_END=${TASK0_WEIGHT_END:-0.10}
export TASK1_WEIGHT_START=${TASK1_WEIGHT_START:-0.35}
export TASK1_WEIGHT_END=${TASK1_WEIGHT_END:-0.25}
export TASK2_WEIGHT_START=${TASK2_WEIGHT_START:-0.20}
export TASK2_WEIGHT_END=${TASK2_WEIGHT_END:-0.30}
export TASK3_WEIGHT_START=${TASK3_WEIGHT_START:-0.10}
export TASK3_WEIGHT_END=${TASK3_WEIGHT_END:-0.20}
export TASK4_WEIGHT_START=${TASK4_WEIGHT_START:-0.10}
export TASK4_WEIGHT_END=${TASK4_WEIGHT_END:-0.15}

# ── Save / eval (recipe §3.4 + §5: loss eval every 1000; delta checkpoints
#    keep only the ~126M/202M trainable Metis weights instead of a full model
#    dump — set CHECKPOINT_SAVE_MODE=full for legacy from_pretrained dumps) ──
export SAVE_STEPS=${SAVE_STEPS:-2000}
export CHECKPOINT_SAVE_MODE=${CHECKPOINT_SAVE_MODE:-delta}
export EVAL_STEPS=${EVAL_STEPS:-1000}
export GEN_EVAL_STEPS=${GEN_EVAL_STEPS:-0}
export EVAL_SAMPLES=${EVAL_SAMPLES:-60}
export EVAL_SAMPLES_PER_TASK=${EVAL_SAMPLES_PER_TASK:-20}

# ── Wandb / output ──────────────────────────────────────────────────
export USE_WANDB=${USE_WANDB:-1}
export WANDB_PROJECT=${WANDB_PROJECT:-metis_training}
export OUTPUT_DIR
export NAME
export RUN_NAME=${NAME}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${NAME}}
LOG_NAME=${LOG_NAME:-${NAME}}

LOG_FILE=${LOG_FILE:-logs/${LOG_NAME}.log}
PID_FILE=${PID_FILE:-logs/${LOG_NAME}.pid}

echo "════════════════════════════════════════════════════════════"
echo " Metis training: ${NAME}"
echo "   model=${MODEL_PATH}"
echo "   backbone_type=${BACKBONE_TYPE}"
echo "   data_format=${DATA_FORMAT}"
echo "   train_data=${TRAIN_DATA}"
echo "   valid_data=${VALID_DATA}"
echo "   memory=${METIS_BLOCK_TYPE}/${METIS_HYPER_MEMORY_TYPE}/${METIS_LOCAL_MEMORY_TYPE}"
echo "   deepspeed=${DS_CONFIG:-off}"
echo "   gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "   bs=${BATCH_SIZE} accum=${GRAD_ACCUM} effective=$((NPROC_PER_NODE*BATCH_SIZE*GRAD_ACCUM)) lr=${LR}"
echo "   epochs=${NUM_EPOCHS} max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} ckpt_mode=${CHECKPOINT_SAVE_MODE}"
echo "   eval_steps=${EVAL_STEPS}(loss) gen_eval_steps=${GEN_EVAL_STEPS}(generation; 0=every eval)"
echo "   output=${OUTPUT_DIR}/${NAME}"
echo "   wandb_run=${WANDB_RUN_NAME}"
echo "   log=${LOG_FILE}"
echo "════════════════════════════════════════════════════════════"
if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN=1, not launching."; exit 0; fi

mkdir -p "${OUTPUT_DIR}" logs
setsid bash scripts/run_train.sh "${EXTRA_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" | tee "${PID_FILE}"
echo "Started pid=${PID} — tail -f ${LOG_FILE}"
