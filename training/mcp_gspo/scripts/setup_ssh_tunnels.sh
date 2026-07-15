#!/usr/bin/env bash
# SSH Tunnel Manager for MCP Server Access
# Usage: setup_ssh_tunnels.sh [start|status|stop]

# Note: avoid set -e because individual SSH failures are reported and counted.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_STANDALONE_ROOT="${RL_STANDALONE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TOPOLOGY_FILE="${MCP_TOPOLOGY_FILE:-$RL_STANDALONE_ROOT/config/mcp_topology.sh}"

DSW_HOSTS=()
DSW_INSTANCES=()
DSW_TMUX_SESSIONS=()
SSH_USER="${SSH_USER:-root}"
SSH_PORT=22
REMOTE_PORT_START=8000
REMOTE_PORT_END=8007
PORTS_PER_INSTANCE=$(( REMOTE_PORT_END - REMOTE_PORT_START + 1 ))
LOCAL_PORT_BASE=18000
PID_DIR="/tmp/mcp_ssh_tunnels"
SSH_STRICT_HOST_KEY_CHECKING="accept-new"

if [[ ! -f "$TOPOLOGY_FILE" ]]; then
    echo "ERROR: topology file not found: $TOPOLOGY_FILE" >&2
    echo "Create it from config/mcp_topology.sh.example." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$TOPOLOGY_FILE"

if (( ${#DSW_HOSTS[@]} == 0 )); then
    echo "ERROR: DSW_HOSTS is empty in $TOPOLOGY_FILE" >&2
    exit 1
fi
if (( ${#DSW_HOSTS[@]} != ${#DSW_INSTANCES[@]} || ${#DSW_HOSTS[@]} != ${#DSW_TMUX_SESSIONS[@]} )); then
    echo "ERROR: DSW_HOSTS, DSW_INSTANCES, and DSW_TMUX_SESSIONS must have equal lengths" >&2
    exit 1
fi

mkdir -p "$PID_DIR"

get_local_port() {
    local instance_idx=$1
    local port_idx=$2
    echo $((LOCAL_PORT_BASE + instance_idx * PORTS_PER_INSTANCE + port_idx))
}

start_tunnels() {
    echo "=========================================="
    echo "Setting up SSH tunnels for MCP servers"
    echo "=========================================="
    echo "DSW instances: ${#DSW_INSTANCES[@]}"
    echo "Ports per instance: ${PORTS_PER_INSTANCE} (${REMOTE_PORT_START}-${REMOTE_PORT_END})"
    echo "Total tunnels: $((${#DSW_INSTANCES[@]} * PORTS_PER_INSTANCE))"
    local last_port=$(get_local_port $((${#DSW_INSTANCES[@]} - 1)) $((PORTS_PER_INSTANCE - 1)))
    echo "Local port range: ${LOCAL_PORT_BASE} - ${last_port}"
    echo "=========================================="
    echo ""

    local success=0
    local fail=0

    for i in "${!DSW_INSTANCES[@]}"; do
        local instance="${DSW_INSTANCES[$i]}"
        local ssh_host="${DSW_HOSTS[$i]}"

        # Build port forwarding arguments
        local port_args=()
        for j in $(seq 0 $((PORTS_PER_INSTANCE - 1))); do
            local local_port=$(get_local_port $i $j)
            local remote_port=$((REMOTE_PORT_START + j))
            port_args+=("-L" "${local_port}:localhost:${remote_port}")
        done

        local first_port=$(get_local_port $i 0)
        local last_inst_port=$(get_local_port $i $((PORTS_PER_INSTANCE - 1)))

        local pid_file="$PID_DIR/tunnel_${instance}.pid"

        # Kill existing tunnel if any (tracked by PID file)
        if [ -f "$pid_file" ]; then
            local old_pid=$(cat "$pid_file")
            if kill -0 "$old_pid" 2>/dev/null; then
                kill "$old_pid" 2>/dev/null || true
                sleep 0.5
            fi
            rm -f "$pid_file"
        fi

        # Refuse to kill unrelated processes occupying the requested range.
        local occupied=0
        for j in $(seq 0 $((PORTS_PER_INSTANCE - 1))); do
            local lp=$(get_local_port $i $j)
            if command -v lsof >/dev/null 2>&1 && lsof -ti ":${lp}" >/dev/null 2>&1; then
                echo "Local port ${lp} is already in use; refusing to replace its owner." >&2
                occupied=1
            fi
        done
        if (( occupied )); then
            echo "[$((i+1))/${#DSW_INSTANCES[@]}] ${instance}... FAILED (port conflict)"
            fail=$((fail+1))
            continue
        fi

        # Start SSH tunnel (background, no shell)
        echo -n "[$((i+1))/${#DSW_INSTANCES[@]}] ${instance} (${PORTS_PER_INSTANCE} ports)... "
        ssh -o "StrictHostKeyChecking=${SSH_STRICT_HOST_KEY_CHECKING}" \
            -o BatchMode=yes \
            -o ConnectTimeout=15 \
            -o ServerAliveInterval=30 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes \
            "${port_args[@]}" \
            -f -N \
            "${SSH_USER}@${ssh_host}" -p "${SSH_PORT}" 2>/dev/null

        if [ $? -eq 0 ]; then
            # Find the SSH process PID. The actual command line contains the
            # SSH host, not the logical instance name, so match by host.
            # ssh -f daemonizes via fork; the surviving process keeps the full
            # command line including `-f -N root@<host>`.
            local pid=$(pgrep -f "ssh.*-f -N ${SSH_USER}@${ssh_host}( |$)" 2>/dev/null | tail -1)
            if [ -n "$pid" ]; then
                echo "$pid" > "$pid_file"
                echo "OK (pid=$pid, local ports: ${first_port}-${last_inst_port})"
                success=$((success+1))
            else
                echo "OK (pid unknown, local ports: ${first_port}-${last_inst_port})"
                success=$((success+1))
            fi
        else
            echo "FAILED"
            fail=$((fail+1))
        fi
    done

    echo ""
    echo "=========================================="
    echo "Result: ${success} OK, ${fail} FAILED"
    echo "=========================================="
    echo ""

    # --- Write MCP_SERVER_URLS (flat list of all URLs) ---
    local all_urls=""
    for i in "${!DSW_INSTANCES[@]}"; do
        for j in $(seq 0 $((PORTS_PER_INSTANCE - 1))); do
            local lp=$(get_local_port $i $j)
            if [ -n "$all_urls" ]; then
                all_urls="${all_urls},"
            fi
            all_urls="${all_urls}http://localhost:${lp}/sse"
        done
    done

    local env_file="$PID_DIR/mcp_server_urls.env"
    cat > "$env_file" << ENVEOF
# Auto-generated by setup_ssh_tunnels.sh at $(date)
# Flat URL list (all DSW instances × ${PORTS_PER_INSTANCE} ports)
export MCP_SERVER_URLS="${all_urls}"

# DSW topology: instance_count, ports_per_instance, local_port_base
export MCP_DSW_COUNT=${#DSW_INSTANCES[@]}
export MCP_PORTS_PER_DSW=${PORTS_PER_INSTANCE}
export MCP_LOCAL_PORT_BASE=${LOCAL_PORT_BASE}

# Restart-on-timeout topology (consumed by MCPServerPool.restart_remote_port)
# Order MUST match DSW_INSTANCES.
export MCP_DSW_HOSTS="$(IFS=,; echo "${DSW_HOSTS[*]}")"
export MCP_DSW_TMUX_SESSIONS="$(IFS=,; echo "${DSW_TMUX_SESSIONS[*]}")"
export MCP_SSH_USER="${SSH_USER}"
export MCP_REMOTE_PORT_BASE=${REMOTE_PORT_START}
ENVEOF

    echo "Environment file: $env_file"
    echo "  source $env_file"
    echo ""
    echo "DSW topology:"
    echo "  ${#DSW_INSTANCES[@]} instances × ${PORTS_PER_INSTANCE} ports = $((${#DSW_INSTANCES[@]} * PORTS_PER_INSTANCE)) total endpoints"
    for i in "${!DSW_INSTANCES[@]}"; do
        local fp=$(get_local_port $i 0)
        local lp=$(get_local_port $i $((PORTS_PER_INSTANCE - 1)))
        echo "  DSW[$i] ${DSW_INSTANCES[$i]}: localhost:${fp}-${lp}"
    done
    echo ""
    echo "Quick test: curl -s -m 3 http://localhost:${LOCAL_PORT_BASE}/sse"
}

check_status() {
    echo "SSH Tunnel Status:"
    echo "=========================================="
    local alive=0
    local dead=0

    for i in "${!DSW_INSTANCES[@]}"; do
        local instance="${DSW_INSTANCES[$i]}"
        local pid_file="$PID_DIR/tunnel_${instance}.pid"
        local fp=$(get_local_port $i 0)
        local lp=$(get_local_port $i $((PORTS_PER_INSTANCE - 1)))

        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  [ALIVE] DSW[$i] ${instance} (pid=$pid, ports: ${fp}-${lp})"
                alive=$((alive+1))
            else
                echo "  [DEAD]  DSW[$i] ${instance} (pid=$pid stale)"
                dead=$((dead+1))
            fi
        else
            echo "  [NONE]  DSW[$i] ${instance}"
            dead=$((dead+1))
        fi
    done

    echo "=========================================="
    echo "Alive: $alive / ${#DSW_INSTANCES[@]}"
}

stop_tunnels() {
    echo "Stopping all SSH tunnels..."
    local killed=0

    # Method 1: PID files (preferred)
    for i in "${!DSW_INSTANCES[@]}"; do
        local instance="${DSW_INSTANCES[$i]}"
        local pid_file="$PID_DIR/tunnel_${instance}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null && killed=$((killed+1))
                echo "  Stopped via pid file: ${instance} (pid=$pid)"
            fi
            rm -f "$pid_file"
        fi
    done

    # Method 2: Match by host (covers tunnels that started without a pid file,
    # e.g. legacy launches before the pgrep fix). The `-f -N root@HOST` token
    # is stable across all our ssh invocations.
    for host in "${DSW_HOSTS[@]}"; do
        local pids
        pids=$(ps -eo pid,args 2>/dev/null \
            | awk -v h="root@${host}" '$0 ~ "ssh" && $0 ~ "-f -N" && $0 ~ h && $0 !~ "awk " {print $1}')
        for p in $pids; do
            if kill "$p" 2>/dev/null; then
                killed=$((killed+1))
                echo "  Stopped via host match: pid=$p (root@${host})"
            fi
        done
    done

    # Method 3: Last-resort port-range match (handles tunnels to hosts no
    # longer in DSW_HOSTS — e.g. you removed an instance and want to clean up).
    local pids
    pids=$(ps -eo pid,args 2>/dev/null \
        | awk -v base="${LOCAL_PORT_BASE}" '
            $0 ~ "ssh" && $0 ~ "-f -N" && $0 ~ ("-L "base) && $0 !~ "awk " {print $1}')
    for p in $pids; do
        if kill "$p" 2>/dev/null; then
            killed=$((killed+1))
            echo "  Stopped via port-base match: pid=$p"
        fi
    done

    echo "Total killed: ${killed}"
    echo "Done."
}

# Main
case "${1:-start}" in
    start)
        start_tunnels
        ;;
    status)
        check_status
        ;;
    stop)
        stop_tunnels
        ;;
    *)
        echo "Usage: $0 {start|status|stop}"
        exit 1
        ;;
esac
