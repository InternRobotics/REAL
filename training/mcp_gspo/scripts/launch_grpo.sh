#!/usr/bin/env bash
set -euo pipefail

RL_STANDALONE_ROOT="${RL_STANDALONE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$RL_STANDALONE_ROOT/scripts"
ENV_FILE="${RL_TRAINING_ENV_FILE:-$RL_STANDALONE_ROOT/config/rl_training.env}"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi

DATE_SUFFIX="${1:-$(date +%m%d)}"
LOG_DIR="${LOG_DIR:-$RL_STANDALONE_ROOT/logs}"
RUN_DIR="${RL_RUN_DIR:-$RL_STANDALONE_ROOT/.run}"
ROLLOUT_LOG="${ROLLOUT_LOG:-$LOG_DIR/debug_rollout_server${DATE_SUFFIX}.log}"
TRAINER_LOG="${TRAINER_LOG:-$LOG_DIR/debug_grpo_trainer${DATE_SUFFIX}.log}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8005}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "========================================"
echo "  GRPO Launch - suffix: ${DATE_SUFFIX}"
echo "========================================"
echo "  Root        : $RL_STANDALONE_ROOT"
echo "  Rollout log : $ROLLOUT_LOG"
echo "  Trainer log : $TRAINER_LOG"
echo "  Rollout port: $ROLLOUT_PORT"
echo "========================================"

for f in "$ROLLOUT_LOG" "$TRAINER_LOG"; do
    if [[ -f "$f" ]]; then
        echo "WARNING: $f already exists; new output will append."
    fi
done

if command -v lsof >/dev/null 2>&1 && lsof -ti :"$ROLLOUT_PORT" >/dev/null 2>&1; then
    echo "ERROR: Port $ROLLOUT_PORT is already in use. Run:"
    echo "  bash $SCRIPT_DIR/kill_grpo_stack.sh"
    exit 1
fi

echo ""
echo "[1/3] Starting rollout server..."
nohup setsid bash "$SCRIPT_DIR/start_rollout_server.sh" >> "$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
echo "$ROLLOUT_PID" > "$RUN_DIR/rollout.pid"
echo "  PID: $ROLLOUT_PID"
echo "  Log: $ROLLOUT_LOG"

echo ""
echo "[2/3] Waiting for rollout server on :${ROLLOUT_PORT} (timeout ${ROLLOUT_STARTUP_TIMEOUT:-300}s)..."
WAIT_START=$(date +%s)
READY=0
while true; do
    if command -v lsof >/dev/null 2>&1 && lsof -ti :"$ROLLOUT_PORT" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        rm -f "$RUN_DIR/rollout.pid"
        echo "ERROR: Rollout server process ($ROLLOUT_PID) died. Check log:"
        echo "  tail -50 $ROLLOUT_LOG"
        exit 1
    fi
    NOW=$(date +%s)
    ELAPSED=$(( NOW - WAIT_START ))
    if (( ELAPSED >= ${ROLLOUT_STARTUP_TIMEOUT:-300} )); then
        bash "$SCRIPT_DIR/kill_grpo_stack.sh" || true
        echo "ERROR: Rollout server did not come up within ${ROLLOUT_STARTUP_TIMEOUT:-300}s. Check log:"
        echo "  tail -50 $ROLLOUT_LOG"
        exit 1
    fi
    printf "\r  Waiting... %ds elapsed" "$ELAPSED"
    sleep 5
done
echo ""
echo "  Rollout server ready after $(( $(date +%s) - WAIT_START ))s"

echo ""
echo "[3/3] Starting GRPO trainer..."
nohup setsid bash "$SCRIPT_DIR/start_grpo_trainer.sh" >> "$TRAINER_LOG" 2>&1 &
TRAINER_PID=$!
echo "$TRAINER_PID" > "$RUN_DIR/trainer.pid"
echo "  PID: $TRAINER_PID"
echo "  Log: $TRAINER_LOG"

echo ""
echo "========================================"
echo "  Both processes launched."
echo ""
echo "  Monitor rollout:  tail -f $ROLLOUT_LOG"
echo "  Monitor trainer:  tail -f $TRAINER_LOG"
echo ""
echo "  Kill all:         bash $SCRIPT_DIR/kill_grpo_stack.sh"
echo ""
echo "  Rollout PID: $ROLLOUT_PID"
echo "  Trainer PID: $TRAINER_PID"
echo "  Ready flag: $READY"
echo "========================================"
