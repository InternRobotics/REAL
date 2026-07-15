#!/usr/bin/env bash

rl_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

rl_load_environment() {
    RL_STANDALONE_ROOT="${RL_STANDALONE_ROOT:-$(rl_root)}"
    export RL_STANDALONE_ROOT

    local env_file="${RL_TRAINING_ENV_FILE:-$RL_STANDALONE_ROOT/config/rl_training.env}"
    if [[ -f "$env_file" ]]; then
        # shellcheck source=/dev/null
        source "$env_file"
    fi

    local conda_sh="${CONDA_SH:-/root/miniconda3/bin/activate}"
    local conda_env="${CONDA_ENV_NAME-mcp-grpo-rl}"
    if [[ -n "$conda_env" ]]; then
        [[ -f "$conda_sh" ]] || {
            echo "ERROR: CONDA_ENV_NAME is set but CONDA_SH does not exist: $conda_sh" >&2
            return 1
        }
        # shellcheck source=/dev/null
        set +u
        source "$conda_sh"
        conda activate "$conda_env"
        set -u
    fi

    export no_proxy="${no_proxy:-localhost,127.0.0.1}"
    export NO_PROXY="${NO_PROXY:-$no_proxy}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export PYTHONPATH="$RL_STANDALONE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

rl_configure_nccl() {
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
    export NCCL_NET="${NCCL_NET:-Socket}"
    unset NCCL_MIN_NCHANNELS
}

rl_configure_cuda_toolkit() {
    local cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
    if [[ -d "$cuda_home" ]]; then
        export CUDA_HOME="$cuda_home"
        export PATH="$cuda_home/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_home/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

rl_visible_gpu_count() {
    local devices="${1:-}"
    [[ -n "$devices" ]] || {
        echo 0
        return
    }
    local compact="${devices//[[:space:]]/}"
    local commas="${compact//[^,]/}"
    echo $(( ${#commas} + 1 ))
}

rl_require_file() {
    local label="$1"
    local path="$2"
    [[ -f "$path" ]] || {
        echo "ERROR: $label does not exist: $path" >&2
        return 1
    }
}

rl_run() {
    if [[ "${RL_DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}
