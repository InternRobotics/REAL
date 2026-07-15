#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/lib/runtime.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/runtime.sh"
rl_load_environment
rl_configure_nccl
rl_configure_cuda_toolkit

export MCP_MAX_CONCURRENT="${MCP_MAX_CONCURRENT:-48}"
export CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

: "${MODEL_PATH:?Set MODEL_PATH in config/rl_training.env or export it before launching.}"

VISIBLE_GPU_COUNT="$(rl_visible_gpu_count "$CUDA_VISIBLE_DEVICES")"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$VISIBLE_GPU_COUNT}"
if [[ "$TENSOR_PARALLEL_SIZE" -ne "$VISIBLE_GPU_COUNT" ]]; then
    echo "ERROR: TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE but CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT device(s)" >&2
    exit 1
fi
PLUGIN_PATH="${PLUGIN_PATH:-$RL_STANDALONE_ROOT/scripts/mcp_rl_env.py}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8005}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SSH_TUNNEL_ENV="${SSH_TUNNEL_ENV:-/tmp/mcp_ssh_tunnels/mcp_server_urls.env}"
TUNNEL_SCRIPT="${TUNNEL_SCRIPT:-$RL_STANDALONE_ROOT/scripts/setup_ssh_tunnels.sh}"
AUTOSTART_TUNNELS="${AUTOSTART_TUNNELS:-0}"

if [[ -z "${MCP_SERVER_URLS:-}" && ! -f "$SSH_TUNNEL_ENV" && "$AUTOSTART_TUNNELS" == "1" && -f "$TUNNEL_SCRIPT" ]]; then
    echo "SSH tunnel env not found; running $TUNNEL_SCRIPT"
    bash "$TUNNEL_SCRIPT"
fi

if [[ -f "$SSH_TUNNEL_ENV" ]]; then
    echo "Loading MCP config from SSH tunnel env: $SSH_TUNNEL_ENV"
    # shellcheck source=/dev/null
    source "$SSH_TUNNEL_ENV"
fi

export MCP_SERVER_URLS="${MCP_SERVER_URLS:-${MCP_SERVER_URL:-http://127.0.0.1:8080/sse}}"
export MCP_STRICT_REAL_CONTRACT="${MCP_STRICT_REAL_CONTRACT:-1}"

export MCP_COMPRESS_HISTORY="${MCP_COMPRESS_HISTORY:-1}"
export MCP_COMPRESS_KEEP_LAST_K="${MCP_COMPRESS_KEEP_LAST_K:-1}"
export MCP_COMPRESS_LAST_OBS_MAX_CHARS="${MCP_COMPRESS_LAST_OBS_MAX_CHARS:-200}"

echo "=========================================="
echo "Starting Rollout Server"
echo "=========================================="
echo "Root: $RL_STANDALONE_ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Plugin: $PLUGIN_PATH"
echo "Port: $ROLLOUT_PORT"
echo "MCP DSW Count: ${MCP_DSW_COUNT:-unknown}"
echo "MCP Ports per DSW: ${MCP_PORTS_PER_DSW:-unknown}"
echo "MCP_SERVER_URLS length: $(echo "$MCP_SERVER_URLS" | tr ',' '\n' | wc -l) endpoints"
echo "History compression: enabled=${MCP_COMPRESS_HISTORY}, keep_last_k=${MCP_COMPRESS_KEEP_LAST_K}, last_obs_max_chars=${MCP_COMPRESS_LAST_OBS_MAX_CHARS}"
echo "=========================================="

rollout_args=(
    -m swift.cli.rollout \
    --model "$MODEL_PATH" \
    --model_type "${MODEL_TYPE:-qwen3_vl}" \
    --external_plugins "$PLUGIN_PATH" \
    --use_gym_env true \
    --gym_env "${GYM_ENV:-mcp}" \
    --multi_turn_scheduler "${MULTI_TURN_SCHEDULER:-mcp_scheduler}" \
    --max_turns "${MAX_TURNS:-50}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-48}" \
    --vllm_tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}" \
    --vllm_enable_lora "${VLLM_ENABLE_LORA:-true}" \
    --vllm_max_lora_rank "${VLLM_MAX_LORA_RANK:-128}" \
    --port "$ROLLOUT_PORT"
)

rl_run "$PYTHON_BIN" "${rollout_args[@]}"
