#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${RL_STANDALONE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RL_RUN_DIR:-$ROOT_DIR/.run}"

stop_group() {
    local name="$1"
    local pid_file="$RUN_DIR/${name}.pid"
    [[ -f "$pid_file" ]] || {
        echo "[$name] no PID file"
        return
    }

    local pid
    pid="$(<"$pid_file")"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "[$name] invalid PID file: $pid_file" >&2
        rm -f "$pid_file"
        return
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[$name] process $pid is already stopped"
        rm -f "$pid_file"
        return
    fi

    local command
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$command" != *"$ROOT_DIR"* ]]; then
        echo "[$name] refusing to stop PID $pid; command is not owned by this checkout: $command" >&2
        return 1
    fi

    echo "[$name] stopping process group $pid"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "[$name] TERM timeout; sending KILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
}

stop_group trainer
stop_group rollout
