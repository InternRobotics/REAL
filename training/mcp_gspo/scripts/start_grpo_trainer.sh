#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/lib/runtime.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/runtime.sh"
rl_load_environment
rl_configure_nccl
rl_configure_cuda_toolkit

export CUDA_VISIBLE_DEVICES="${TRAINER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${MODEL_PATH:?Set MODEL_PATH in config/rl_training.env or export it before launching.}"

VISIBLE_GPU_COUNT="$(rl_visible_gpu_count "$CUDA_VISIBLE_DEVICES")"
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPU_COUNT}"
if [[ "$NUM_GPUS" -ne "$VISIBLE_GPU_COUNT" ]]; then
    echo "ERROR: NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT device(s)" >&2
    exit 1
fi
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
DATASET_PATH="${DATASET_PATH:-$RL_STANDALONE_ROOT/scripts/grpo_gym_dataset.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RL_STANDALONE_ROOT/output/grpo_mcp}"
ROLLOUT_HOST="${ROLLOUT_HOST:-localhost}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8005}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-ngrpo}"

mkdir -p "$OUTPUT_DIR/logs"
rl_require_file "dataset" "$DATASET_PATH"

if [[ "$ADVANTAGE_ESTIMATOR" == "ngrpo" ]]; then
    if ! "$PYTHON_BIN" "$RL_STANDALONE_ROOT/scripts/patch_ms_swift_ngrpo.py"; then
        echo "ERROR: NGRPO requires the local ms-swift reroute." >&2
        echo "Run: $PYTHON_BIN $RL_STANDALONE_ROOT/scripts/patch_ms_swift_ngrpo.py --apply" >&2
        exit 1
    fi
fi

extra_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    extra_args+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "=========================================="
echo "Starting GRPO Trainer"
echo "=========================================="
echo "Root: $RL_STANDALONE_ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Num GPUs: $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Rollout Server: $ROLLOUT_HOST:$ROLLOUT_PORT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

trainer_args=(
    -m swift.cli.rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --model_type "${MODEL_TYPE:-qwen3_vl}" \
    --train_type "${TRAIN_TYPE:-lora}" \
    --lora_rank "${LORA_RANK:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --target_modules "${TARGET_MODULES:-all-linear}" \
    --dataset "$DATASET_PATH" \
    --use_vllm true \
    --vllm_mode server \
    --vllm_enable_lora true \
    --vllm_server_host "$ROLLOUT_HOST" \
    --vllm_server_port "$ROLLOUT_PORT" \
    --num_generations "${NUM_GENERATIONS:-8}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-2048}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "${MAX_STEPS:-1000}" \
    --learning_rate "${LEARNING_RATE:-5e-6}" \
    --warmup_ratio "${WARMUP_RATIO:-0.0}" \
    --freeze_vit "${FREEZE_VIT:-true}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
    --beta "${BETA:-0.04}" \
    --log_entropy "${LOG_ENTROPY:-true}" \
    --report_to "${REPORT_TO:-tensorboard}" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --advantage_estimator "$ADVANTAGE_ESTIMATOR" \
    --logging_steps "${LOGGING_STEPS:-1}" \
    --save_steps "${SAVE_STEPS:-10}" \
    --max_length "${MAX_LENGTH:-8192}" \
    --importance_sampling_level "${IMPORTANCE_SAMPLING_LEVEL:-sequence}"
)

if [[ -n "${NGRPO_VIRTUAL_MAX_REWARD:-}" ]]; then
    trainer_args+=(--ngrpo_virtual_max_reward "$NGRPO_VIRTUAL_MAX_REWARD")
fi

if [[ "${DEEPSPEED_STAGE:-zero2}" != "none" ]]; then
    trainer_args+=(--deepspeed "${DEEPSPEED_STAGE:-zero2}")
fi
trainer_args+=("${extra_args[@]}")

rl_run "$TORCHRUN_BIN" --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" "${trainer_args[@]}"
