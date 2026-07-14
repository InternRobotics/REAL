#!/usr/bin/env bash
set -euo pipefail

# Demo MCP Server — non-headless (opens Omniverse GUI window)
# For interactive demo with visual inspection.
#
# Usage:
#   ./run_mcp_server_demo.sh
#
# Optional overrides:
#   DEMO_TASK_CONFIG=configs/demo_task.yaml  — task config (default shown)
#   TARGET_SCENE_ID=...                      — override scene_id in config
#   HOST=127.0.0.1                           — bind host (default localhost)
#   PORT=8080                                — server port (default 8080)
#   IS_TEST=0                                — test mode flag
#   TRAJ_PATH=eval_output_demo               — output directory
#   USE_LIFT_ROBOT=0  LIFT_USD_PATH=...      — enable lift robot
#   USE_EMPTY_SCENE=0 EMPTY_USD_PATH=...     — use empty scene instead

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export DEMO_TASK_CONFIG="${DEMO_TASK_CONFIG:-${REPO_ROOT}/configs/demo_task.yaml}"
export TRAJ_PATH="${TRAJ_PATH:-${REPO_ROOT}/eval_output_demo}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8080}"
export IS_TEST="${IS_TEST:-0}"

echo "Starting MCP Server (DEMO / non-headless)..."
echo "Config:   ${DEMO_TASK_CONFIG}"
echo "Endpoint: http://${HOST}:${PORT}/sse"
echo "IS_TEST:  ${IS_TEST}"
echo ""

cd "${REPO_ROOT}"
python -m mcp_server.mcp_server_demo
