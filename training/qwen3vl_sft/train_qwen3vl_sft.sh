#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim's conda activation exports a synthetic BASH_SOURCE value.  This
# launcher is executed (not sourced), so $0 is the reliable script location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${QWEN3VL_FINETUNE_ROOT:?Set QWEN3VL_FINETUNE_ROOT to the official Qwen3-VL/qwen-vl-finetune directory.}"

if [[ ! -f "${QWEN3VL_FINETUNE_ROOT}/qwenvl/train/train_qwen.py" ]]; then
    echo "Could not find qwenvl/train/train_qwen.py under QWEN3VL_FINETUNE_ROOT=${QWEN3VL_FINETUNE_ROOT}" >&2
    exit 1
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASETS="${DATASETS:-real_basic_pnp}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen3vl_sft}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/deepspeed/zero2.json}"
RUN_NAME="${RUN_NAME:-real_qwen3vl_sft}"

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

if [[ "${DEEPSPEED_CONFIG}" != /* ]]; then
    DEEPSPEED_CONFIG="${REPO_ROOT}/${DEEPSPEED_CONFIG}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-784}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
REPORT_TO="${REPORT_TO:-none}"
WARMUP_RATIO="${WARMUP_RATIO:-0.08}"

mkdir -p "${OUTPUT_DIR}"

cd "${QWEN3VL_FINETUNE_ROOT}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    qwenvl/train/train_qwen.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --dataset_use "${DATASETS}" \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_pixels "${MAX_PIXELS}" \
    --min_pixels "${MIN_PIXELS}" \
    --eval_strategy "no" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0 \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --run_name "${RUN_NAME}" \
    --report_to "${REPORT_TO}"
