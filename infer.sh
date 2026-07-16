#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    CHECKPOINT_PATH=$1
    shift
else
    CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
fi
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "Usage: bash infer.sh CHECKPOINT_PATH [run_inference.py arguments...]" >&2
    echo "   or: CHECKPOINT_PATH=/path/to/checkpoint bash infer.sh [arguments...]" >&2
    exit 2
fi

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}

ARGS=(
    --checkpoint_path "${CHECKPOINT_PATH}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
)

if [ -n "${MODEL_PATH:-}" ]; then
    ARGS+=(--model_path "${MODEL_PATH}")
fi
if [ -n "${TOKENIZER_PATH:-}" ]; then
    ARGS+=(--tokenizer_path "${TOKENIZER_PATH}")
fi

exec "${PYTHON}" run_inference.py "${ARGS[@]}" "$@"
