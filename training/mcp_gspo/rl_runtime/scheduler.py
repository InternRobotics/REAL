"""ms-swift multi-turn scheduler for REAL MCP rollouts."""

import asyncio
import json
import math
import os
import random
import sys
import time

from rl_runtime.environment import (
    HistoryManager,
    MCPSingleEnv,
    MCPSwiftEnv,
    _runtime_path,
    should_treat_reset_error_as_task_not_found,
)


# ============================================================
# MCPMultiTurnScheduler - Single-turn inference for each step
# ============================================================
# This scheduler implements run() for single-turn VLM models:
# - Each step is an INDEPENDENT single-turn inference (no conversation history)
# - Task progress is encoded in the prompt template, not in chat history
# - Same approach as test_rl_env_with_model.py

try:
    from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns, RolloutScheduler
    from swift.plugin.env import Env, envs
    from copy import deepcopy
    from typing import TYPE_CHECKING, Any, Union, List, Dict, Optional

    if TYPE_CHECKING:
        from swift.llm.infer.protocol import ChatCompletionResponseChoice, RolloutOutput, RequestConfig
        from swift.llm.template import RolloutInferRequest

    # Keep scheduler prompt fully aligned with MCPSwiftEnv training prompt.
    # This ensures ask constraints and hand-state context are visible during rollout.
    MCP_PROMPT_TEMPLATE = MCPSwiftEnv.PROMPT_TEMPLATE
    MCP_COMPRESSED_HISTORY_TEMPLATE = (
        "This is a simplified history state summary. "
        "You are a tool calling embodied agent, your task is {task}, latest progress step is {latest_progress_step}. "
        "Your last action is {last_action}. After that, environment observed {last_obs}. "
        'You have called "ask" {ask_count} times, and your hand state is {hand_state_description}.'
    )

    # =========================================================================

    # =========================================================================
    class MCPServerPool:
        _instance = None
        _lock = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

        def __init__(self):
            if self._initialized:
                return
            import asyncio
            import threading

            self.dsw_count = int(os.environ.get("MCP_DSW_COUNT", "0"))
            self.ports_per_dsw = int(os.environ.get("MCP_PORTS_PER_DSW", "0"))
            self.local_port_base = int(os.environ.get("MCP_LOCAL_PORT_BASE", "18000"))

            # Restart-on-timeout topology (set by setup_ssh_tunnels.sh).
            # Falls back to empty lists if not provided — restart_remote_port
            # will then refuse to act and the port stays out of rotation.
            self.dsw_hosts = [h.strip() for h in os.environ.get("MCP_DSW_HOSTS", "").split(",") if h.strip()]
            self.dsw_tmux_sessions = [
                s.strip() for s in os.environ.get("MCP_DSW_TMUX_SESSIONS", "").split(",") if s.strip()
            ]
            self.ssh_user = os.environ.get("MCP_SSH_USER", "root")
            self.remote_port_base = int(os.environ.get("MCP_REMOTE_PORT_BASE", "8000"))

            multi_server_env = os.environ.get("MCP_SERVER_URLS", "")
            if multi_server_env:
                all_urls = [url.strip().rstrip("/") for url in multi_server_env.split(",") if url.strip()]
            else:
                all_urls = [os.environ.get("MCP_SERVER_URL", "http://localhost:8080/sse").rstrip("/")]

            if self.dsw_count > 0 and self.ports_per_dsw > 0:
                self.dsw_groups = []  # List of List[str], dsw_groups[i] = URLs for DSW i
                for i in range(self.dsw_count):
                    start = i * self.ports_per_dsw
                    end = start + self.ports_per_dsw
                    group_urls = all_urls[start:end]
                    if group_urls:
                        self.dsw_groups.append(group_urls)
            else:
                self.dsw_groups = [all_urls]

            self.server_urls = all_urls

            self._dsw_queue = None

            self._port_queues = {}  # dsw_index -> asyncio.Queue[url]

            self._failed_servers = set()
            self._checking_servers = set()
            # Permanently dead DSWs (3-retry startup probe exhausted). Health
            # check skips ports in these DSWs so we never try them again.
            self.dead_dsws = set()
            # Set to True once _startup_probe has finished so async_infer can
            # synchronously check "are any DSWs usable at all?" before rollout.
            self._startup_probe_done = None  # asyncio.Event, lazy-init in loop

            self._threading_lock = threading.Lock()
            self._initialized = True
            self._health_check_task = None
            self._health_check_interval = 30.0

            print(f"[MCPServerPool] DSW-aware pool initialized:")
            print(f"  DSW instances: {len(self.dsw_groups)}")
            print(f"  Ports per DSW: {self.ports_per_dsw or len(self.dsw_groups[0]) if self.dsw_groups else 0}")
            print(f"  Total endpoints: {len(all_urls)}")
            for i, group in enumerate(self.dsw_groups):
                print(f"  DSW[{i}]: {len(group)} ports ({group[0]}...{group[-1]})")

        def _ensure_queue(self):
            import asyncio

            if self._dsw_queue is None:
                with self._threading_lock:
                    if self._dsw_queue is None:
                        self._dsw_queue = asyncio.Queue()
                        for dsw_idx, urls in enumerate(self.dsw_groups):
                            self._dsw_queue.put_nowait(dsw_idx)

                            q = asyncio.Queue()
                            for url in urls:
                                q.put_nowait(url)
                            self._port_queues[dsw_idx] = q
                        print(f"[MCPServerPool] Queues initialized: {self._dsw_queue.qsize()} DSW groups")

                        self._start_health_check()
                        # Schedule a startup probe to quickly mark dead DSWs
                        try:
                            loop = asyncio.get_event_loop()
                            self._startup_probe_done = asyncio.Event()
                            loop.create_task(self._startup_probe())
                        except Exception:
                            pass

        async def wait_startup_probe(self, timeout: float = 120.0) -> None:
            """Block until the startup connectivity probe completes (or timeout).
            Safe to call before any DSW-dependent operation. No-op if probe
            was never scheduled (e.g. pool used outside the normal setup)."""
            import asyncio

            if self._startup_probe_done is None:
                return
            try:
                await asyncio.wait_for(self._startup_probe_done.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                print(
                    f"[MCPServerPool] WARNING: startup probe did not finish within {timeout}s, "
                    f"proceeding with current dead_dsws={sorted(self.dead_dsws)}"
                )

        async def _startup_probe(self):
            """Probe every port of every DSW at startup with up to 3 retry rounds.

            - Each round probes ALL ports of ALL (still-pending) DSWs in parallel.
            - A DSW is considered alive as soon as ≥1 of its ports responds.
            - After 3 failed rounds (no port responded in any of them), the DSW
              is added to `self.dead_dsws` and its ports are marked failed; the
              health-check loop will NOT attempt to recover them.
            """
            import asyncio

            PROBE_TIMEOUT = 5.0
            MAX_ATTEMPTS = 3
            RETRY_DELAY = 2.0

            print(
                f"[MCPServerPool] Starting connectivity probe for "
                f"{len(self.dsw_groups)} DSWs (up to {MAX_ATTEMPTS} attempts each)..."
            )

            pending = {dsw_idx: urls for dsw_idx, urls in enumerate(self.dsw_groups)}
            alive_counts: Dict[int, int] = {}

            for attempt in range(1, MAX_ATTEMPTS + 1):
                if not pending:
                    break

                # Probe all ports of all still-pending DSWs in parallel
                probe_spec = [(dsw_idx, url) for dsw_idx, urls in pending.items() for url in urls]
                results = await asyncio.gather(
                    *[self._probe_dsw(dsw_idx, url, PROBE_TIMEOUT) for dsw_idx, url in probe_spec],
                    return_exceptions=True,
                )

                alive_by_dsw: Dict[int, int] = {}
                for (dsw_idx, _url), ok in zip(probe_spec, results):
                    if ok is True:
                        alive_by_dsw[dsw_idx] = alive_by_dsw.get(dsw_idx, 0) + 1

                # Any DSW with ≥1 alive port is resolved this round
                for dsw_idx in list(pending.keys()):
                    if alive_by_dsw.get(dsw_idx, 0) > 0:
                        alive_counts[dsw_idx] = alive_by_dsw[dsw_idx]
                        print(
                            f"[MCPServerPool] DSW[{dsw_idx}] startup probe attempt {attempt}: "
                            f"{alive_by_dsw[dsw_idx]}/{len(pending[dsw_idx])} ports alive"
                        )
                        pending.pop(dsw_idx, None)

                if pending and attempt < MAX_ATTEMPTS:
                    print(
                        f"[MCPServerPool] {len(pending)} DSWs still unreachable "
                        f"({sorted(pending.keys())}), retrying in {RETRY_DELAY}s "
                        f"(attempt {attempt + 1}/{MAX_ATTEMPTS})..."
                    )
                    await asyncio.sleep(RETRY_DELAY)

            # Any DSW still in `pending` has failed all MAX_ATTEMPTS rounds.
            for dsw_idx, urls in pending.items():
                self.dead_dsws.add(dsw_idx)
                # Drain the port queue and mark all ports as failed
                q = self._port_queues.get(dsw_idx)
                if q:
                    while not q.empty():
                        try:
                            url = q.get_nowait()
                            self._failed_servers.add(url)
                        except Exception:
                            break
                for url in urls:
                    self._failed_servers.add(url)
                # Remove this DSW from the DSW queue
                import asyncio as _asyncio

                new_dsw_queue = _asyncio.Queue()
                while not self._dsw_queue.empty():
                    try:
                        idx = self._dsw_queue.get_nowait()
                        if idx != dsw_idx:
                            new_dsw_queue.put_nowait(idx)
                    except Exception:
                        break
                self._dsw_queue = new_dsw_queue
                print(
                    f"[MCPServerPool] DSW[{dsw_idx}] PERMANENTLY DEAD — "
                    f"all {len(urls)} ports failed {MAX_ATTEMPTS} retry rounds; "
                    f"will not be probed or recovered."
                )

            alive = len(alive_counts)
            dead = len(self.dead_dsws)
            print(
                f"[MCPServerPool] Startup probe done: {alive} alive, {dead} dead DSWs (dead={sorted(self.dead_dsws)})"
            )

            if self._startup_probe_done is not None:
                self._startup_probe_done.set()

        async def _probe_dsw(self, dsw_idx: int, url: str, timeout: float) -> bool:
            """Probe one port of a DSW via SSE streaming handshake.

            TCP-only probe is insufficient: SSH tunnel may be alive but remote
            MCP server dead. We stream-GET the SSE endpoint and check if the
            server sends back any data (e.g. 'event: endpoint').
            """
            import httpx

            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", url, timeout=timeout) as resp:
                        async for line in resp.aiter_lines():
                            # Got any response → server is alive
                            print(f"[MCPServerPool] DSW[{dsw_idx}] probe {url}: OK (SSE)")
                            return True
                # Stream ended without any line
                print(f"[MCPServerPool] DSW[{dsw_idx}] probe {url}: FAIL (empty SSE)")
                return False
            except Exception as e:
                print(f"[MCPServerPool] DSW[{dsw_idx}] probe {url}: FAIL ({type(e).__name__})")
                return False

        def _start_health_check(self):
            import asyncio

            async def health_check_loop():
                while True:
                    await asyncio.sleep(self._health_check_interval)
                    if not self._failed_servers:
                        continue
                    servers_to_check = list(self._failed_servers - self._checking_servers)
                    if not servers_to_check:
                        continue
                    print(f"[MCPServerPool] Health check: {len(servers_to_check)} failed servers...")
                    for server_url in servers_to_check:
                        self._checking_servers.add(server_url)
                        try:
                            import httpx

                            recovered = False
                            async with httpx.AsyncClient() as client:
                                async with client.stream("GET", server_url, timeout=5.0) as resp:
                                    async for line in resp.aiter_lines():
                                        recovered = True
                                        break
                            if recovered:
                                # Never resurrect ports that belong to a DSW
                                # declared permanently dead by the startup probe.
                                owning_dsw = next(
                                    (i for i, urls in enumerate(self.dsw_groups) if server_url in urls),
                                    None,
                                )
                                if owning_dsw is not None and owning_dsw in self.dead_dsws:
                                    print(
                                        f"[MCPServerPool] Ignoring recovery of {server_url} "
                                        f"(DSW[{owning_dsw}] is permanently dead)"
                                    )
                                else:
                                    print(f"[MCPServerPool] Server recovered: {server_url}")
                                    self._failed_servers.discard(server_url)

                                    for dsw_idx, urls in enumerate(self.dsw_groups):
                                        if server_url in urls:
                                            await self._port_queues[dsw_idx].put(server_url)
                                            break
                        except Exception as e:
                            print(f"[MCPServerPool] Still down: {server_url} ({type(e).__name__})")
                        finally:
                            self._checking_servers.discard(server_url)

            try:
                loop = asyncio.get_event_loop()
                self._health_check_task = loop.create_task(health_check_loop())
            except Exception as e:
                print(f"[MCPServerPool] Warning: health check start failed: {e}")

        async def acquire_dsw(self, timeout: float = 300.0) -> int:
            import asyncio

            self._ensure_queue()

            available = self._dsw_queue.qsize()
            print(f"[MCPServerPool] Acquiring DSW... (available: {available}/{len(self.dsw_groups)})")

            try:
                dsw_idx = await asyncio.wait_for(self._dsw_queue.get(), timeout=timeout)
                print(f"[MCPServerPool] Acquired DSW[{dsw_idx}] (remaining: {self._dsw_queue.qsize()})")
                return dsw_idx
            except asyncio.TimeoutError:
                print(f"[MCPServerPool] TIMEOUT waiting for DSW after {timeout}s")
                raise

        async def release_dsw(self, dsw_idx: int):
            self._ensure_queue()
            await self._dsw_queue.put(dsw_idx)
            print(f"[MCPServerPool] Released DSW[{dsw_idx}] (available: {self._dsw_queue.qsize()})")

        # =====================================================================
        # In-place remote restart (kill+relaunch the stuck MCP server process
        # via SSH+tmux) so a hung port can be recovered without losing the
        # task_id ↔ DSW pairing GRPO depends on.
        # =====================================================================
        def _local_to_remote_port(self, server_url: str) -> int:
            """Map a local-tunnel URL like http://localhost:18039/sse to remote port 8007."""
            import re

            m = re.search(r":(\d+)(?:/|$)", server_url)
            if not m:
                raise ValueError(f"Cannot parse local port from URL: {server_url}")
            local_port = int(m.group(1))
            j = (local_port - self.local_port_base) % max(self.ports_per_dsw, 1)
            return self.remote_port_base + j

        async def _ssh_run(self, host: str, cmd: str, timeout: float = 30.0):
            """Run a command via SSH, return (returncode, stdout, stderr)."""
            import asyncio

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                f"{self.ssh_user}@{host}",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return -1, "", f"ssh timed out after {timeout}s"

        async def restart_remote_port(self, dsw_idx: int, server_url: str, ready_timeout: float = 300.0) -> bool:
            """SIGTERM (+SIGKILL fallback) the stuck MCP server process and relaunch
            it inside its tmux pane. Polls the remote SSE endpoint until HTTP 200
            or `ready_timeout` elapses. Returns True on success.

            Restart preserves task semantics: the relaunched process loads the
            same scene + task pool from env vars already exported in the tmux
            shell, so (DSW, task_id) -> identical task_description.
            """
            import asyncio

            if dsw_idx >= len(self.dsw_hosts) or dsw_idx >= len(self.dsw_tmux_sessions):
                print(
                    f"[MCPServerPool] No SSH/tmux config for DSW[{dsw_idx}] "
                    f"(hosts={len(self.dsw_hosts)}, sessions={len(self.dsw_tmux_sessions)}); "
                    f"skipping restart of {server_url}"
                )
                return False
            host = self.dsw_hosts[dsw_idx]
            tmux_sess = self.dsw_tmux_sessions[dsw_idx]
            try:
                remote_port = self._local_to_remote_port(server_url)
            except ValueError as e:
                print(f"[MCPServerPool] {e}; skipping restart")
                return False
            win = f"server_{remote_port}"
            tag = f"DSW[{dsw_idx}]:{remote_port}"
            print(f"[MCPServerPool] {tag} restart begin (host={host}, tmux={tmux_sess}:{win})")

            # 1. Find the PID by matching PORT=<remote_port> in /proc/<pid>/environ.
            #    pgrep on cmdline alone is not enough (all 8 processes share argv).
            find_pid_cmd = (
                f"for p in $(pgrep -f mcp_server_debug); do "
                f"  if grep -q '^PORT={remote_port}$' /proc/$p/environ 2>/dev/null; then echo $p; fi; "
                f"done"
            )
            rc, pid_out, err = await self._ssh_run(host, find_pid_cmd, timeout=20.0)
            pid = pid_out.strip().splitlines()[0] if pid_out.strip() else ""
            print(f"[MCPServerPool] {tag} found pid='{pid}' (rc={rc} err={err.strip()[:120]})")

            # 2. SIGTERM, wait up to 12s, then SIGKILL fallback. Isaac Sim does
            #    NOT honor SIGINT (Ctrl-C) but exits cleanly on SIGTERM in <8s.
            if pid and pid.isdigit():
                await self._ssh_run(host, f"kill -TERM {pid} 2>/dev/null; true", timeout=15.0)
                killed = False
                for _ in range(6):
                    await asyncio.sleep(2)
                    rc2, alive, _ = await self._ssh_run(
                        host,
                        f"ps -p {pid} >/dev/null 2>&1 && echo ALIVE || echo DEAD",
                        timeout=15.0,
                    )
                    if "DEAD" in alive:
                        killed = True
                        break
                if not killed:
                    print(f"[MCPServerPool] {tag} pid {pid} survived SIGTERM; sending SIGKILL")
                    await self._ssh_run(host, f"kill -KILL {pid} 2>/dev/null; true", timeout=15.0)
                    await asyncio.sleep(2)

            # 3. Send Ctrl-C to clear any half-typed line from prior interrupt
            #    attempts, then re-issue the launch command. Env vars exported
            #    earlier in this bash session (PORT, CUDA_VISIBLE_DEVICES,
            #    HEADLESS, TARGET_SCENE_ID, ...) persist across the python exit.
            await self._ssh_run(host, f"tmux send-keys -t {tmux_sess}:{win} C-c", timeout=15.0)
            await asyncio.sleep(1)
            await self._ssh_run(
                host,
                f"tmux send-keys -t {tmux_sess}:{win} 'python -m mcp_server.mcp_server_debug' Enter",
                timeout=15.0,
            )
            print(f"[MCPServerPool] {tag} relaunch issued; polling readiness...")

            # 4. Poll readiness via REMOTE curl (local SSH-tunnel returns 503 for
            #    GET /sse without streaming, even when the server is healthy).
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            deadline = t0 + ready_timeout
            probe_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' -m 3 http://localhost:{remote_port}/sse"
            while loop.time() < deadline:
                await asyncio.sleep(10)
                _, code_out, _ = await self._ssh_run(host, probe_cmd, timeout=20.0)
                code = code_out.strip()
                elapsed = loop.time() - t0
                print(f"[MCPServerPool] {tag} probe T+{elapsed:.0f}s -> HTTP '{code}'")
                if code == "200":
                    print(f"[MCPServerPool] {tag} READY after restart ({elapsed:.0f}s)")
                    return True
            print(f"[MCPServerPool] {tag} restart TIMEOUT after {ready_timeout}s")
            return False

        async def restart_and_recycle_port(
            self, dsw_idx: int, server_url: str, port_queue, ready_timeout: float = 300.0
        ):
            """Background helper: restart_remote_port + put the URL back on success.

            On failure, mark URL as failed so the round-port queue gracefully
            shrinks instead of looping forever on a dead port.
            """
            try:
                ok = await self.restart_remote_port(dsw_idx, server_url, ready_timeout=ready_timeout)
            except Exception as e:
                print(f"[MCPServerPool] restart_and_recycle_port({server_url}) raised: {e}")
                ok = False
            if ok:
                try:
                    port_queue.put_nowait(server_url)
                    print(f"[MCPServerPool] {server_url} recycled to queue (size={port_queue.qsize()})")
                except Exception as e:
                    print(f"[MCPServerPool] Could not put {server_url} back: {e}")
                    self._failed_servers.add(server_url)
            else:
                print(f"[MCPServerPool] {server_url} restart failed; marking as failed (port left out of rotation)")
                self._failed_servers.add(server_url)

        async def acquire_port(self, dsw_idx: int, timeout: float = 300.0) -> str:
            import asyncio

            self._ensure_queue()

            q = self._port_queues.get(dsw_idx)
            if q is None:
                raise ValueError(f"Invalid DSW index: {dsw_idx}")

            # Skip already-failed ports — keep dequeuing until we find a good one
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    print(f"[MCPServerPool] TIMEOUT waiting for port on DSW[{dsw_idx}]")
                    raise asyncio.TimeoutError(f"No healthy port on DSW[{dsw_idx}]")
                try:
                    url = await asyncio.wait_for(q.get(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    print(f"[MCPServerPool] TIMEOUT waiting for port on DSW[{dsw_idx}]")
                    raise asyncio.TimeoutError(f"No healthy port on DSW[{dsw_idx}]")
                if url in self._failed_servers:
                    # This port is known-dead; discard it (don't put back)
                    print(f"[MCPServerPool] Skipping known-failed port: {url}")
                    continue
                print(f"[MCPServerPool] Acquired port from DSW[{dsw_idx}]: {url} (remaining: {q.qsize()})")
                return url

        async def release_port(self, dsw_idx: int, server_url: str, failed: bool = False):
            self._ensure_queue()

            if failed:
                if server_url not in self._failed_servers:
                    self._failed_servers.add(server_url)
                    print(f"[MCPServerPool] Port FAILED: {server_url}")
                    # If all ports of this DSW are now failed, log a warning
                    dsw_urls = self.dsw_groups[dsw_idx] if dsw_idx < len(self.dsw_groups) else []
                    failed_in_dsw = sum(1 for u in dsw_urls if u in self._failed_servers)
                    if failed_in_dsw == len(dsw_urls):
                        print(f"[MCPServerPool] WARNING: ALL ports of DSW[{dsw_idx}] are now FAILED")
            else:
                q = self._port_queues.get(dsw_idx)
                if q is not None:
                    await q.put(server_url)

        async def acquire(self, timeout: float = 300.0) -> str:
            import asyncio

            self._ensure_queue()

            # Try non-blocking from any DSW — skip known-failed ports
            for dsw_idx, q in self._port_queues.items():
                while q.qsize() > 0:
                    try:
                        url = q.get_nowait()
                        if url in self._failed_servers:
                            continue  # Discard failed port
                        print(f"[MCPServerPool] Acquired (compat): {url}")
                        return url
                    except asyncio.QueueEmpty:
                        break

            # All busy — wait on any DSW queue that has healthy ports
            print(f"[MCPServerPool] All ports busy, waiting...")
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                for dsw_idx, q in self._port_queues.items():
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        url = await asyncio.wait_for(q.get(), timeout=min(remaining, 2.0))
                        if url in self._failed_servers:
                            continue
                        print(f"[MCPServerPool] Acquired (compat, waited): {url}")
                        return url
                    except asyncio.TimeoutError:
                        continue
            raise asyncio.TimeoutError(f"No server available after {timeout}s")

        async def release(self, server_url: str, failed: bool = False):
            self._ensure_queue()

            if failed:
                if server_url not in self._failed_servers:
                    self._failed_servers.add(server_url)
                    print(f"[MCPServerPool] Server FAILED (compat): {server_url}")
                return

            for dsw_idx, urls in enumerate(self.dsw_groups):
                if server_url in urls:
                    await self._port_queues[dsw_idx].put(server_url)
                    return

            print(f"[MCPServerPool] Warning: unknown server URL: {server_url}")

        def get_dsw_urls(self, dsw_idx: int) -> list:
            if 0 <= dsw_idx < len(self.dsw_groups):
                return self.dsw_groups[dsw_idx]
            return []

        @staticmethod
        def _normalize_task_list(raw) -> list:
            """Normalize a `list_tasks` response (dict/list/mixed) to a list of task_id strings."""
            import re as _re

            _tid_re = _re.compile(r"^(?:task_\d|seed_)")
            if isinstance(raw, dict):
                if "tasks" in raw and isinstance(raw["tasks"], list):
                    inner = raw["tasks"]
                    return [
                        t["task_id"] if isinstance(t, dict) else str(t)
                        for t in inner
                        if (isinstance(t, dict) and "task_id" in t) or isinstance(t, str)
                    ]
                return [k for k in raw.keys() if _tid_re.match(k)]
            if isinstance(raw, list):
                if raw and isinstance(raw[0], dict):
                    return [t["task_id"] for t in raw if "task_id" in t]
                return [str(t) for t in raw if isinstance(t, str)]
            return []

        async def _fetch_task_list_single(self, url: str, timeout: float = 30.0) -> list:
            """Probe one MCP server URL and return its (normalized) task list."""
            from mcp.client.sse import sse_client
            from mcp.client.session import ClientSession

            async with sse_client(url, timeout=timeout) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("list_tasks", {})
                    raw = json.loads(result.content[0].text)
                    return self._normalize_task_list(raw)

        async def fetch_task_lists_per_port(
            self,
            dsw_idx: int,
            verified_urls: list = None,
        ) -> Dict[str, list]:
            """Probe every verified port of a DSW in parallel and return per-port task lists.

            Returns {url: task_list}. URLs whose probe fails are omitted.

            This is used by the scheduler to detect scene heterogeneity within a DSW.
            If ports return different task sets, a task selected from one port may not
            load on another port and can cause an unbounded reset retry loop.
            """
            urls = verified_urls or self.get_dsw_urls(dsw_idx)
            if not urls:
                raise ValueError(f"No URLs for DSW[{dsw_idx}]")

            async def _probe(u):
                try:
                    return u, await self._fetch_task_list_single(u)
                except Exception as e:
                    print(
                        f"[MCPServerPool] DSW[{dsw_idx}] list_tasks on {u} failed: {type(e).__name__}: {str(e)[:120]}"
                    )
                    return u, None

            results = await asyncio.gather(*[_probe(u) for u in urls])
            per_port = {u: tl for u, tl in results if tl is not None}
            print(f"[MCPServerPool] DSW[{dsw_idx}] per-port probe: {len(per_port)}/{len(urls)} ports responded")
            return per_port

        async def fetch_task_list(self, dsw_idx: int, verified_urls: list = None) -> list:
            """Fetch available task_ids from a DSW by calling list_tasks on one of its ports.

            Kept for backward compatibility with paths that only need a single-port view
            (e.g. fallback in the invalid_task resample path). Prefer
            `fetch_task_lists_per_port` + majority-scene filtering for correctness.

            Args:
                dsw_idx: DSW instance index.
                verified_urls: If provided, only try these URLs (already scene-verified).
                    Falls back to get_dsw_urls(dsw_idx) if not provided.
            """
            urls = verified_urls or self.get_dsw_urls(dsw_idx)
            if not urls:
                raise ValueError(f"No URLs for DSW[{dsw_idx}]")

            last_error = None
            for url in urls[:3]:  # Try up to 3 ports
                try:
                    task_list = await self._fetch_task_list_single(url)
                    print(f"[MCPServerPool] DSW[{dsw_idx}] has {len(task_list)} tasks")
                    return task_list
                except Exception as e:
                    last_error = e
                    continue
            raise RuntimeError(f"Failed to fetch task list from DSW[{dsw_idx}]: {last_error}")

        @property
        def num_servers(self) -> int:
            return len(self.server_urls)

        @property
        def num_dsw(self) -> int:
            return len(self.dsw_groups)

        @property
        def available_count(self) -> int:
            if self._dsw_queue is None:
                return len(self.server_urls)
            total = 0
            for q in self._port_queues.values():
                total += q.qsize()
            return total

        @property
        def failed_count(self) -> int:
            return len(self._failed_servers)

    class MCPMultiTurnScheduler(RolloutScheduler):
        """
        MCP environment scheduler with batch-aware DSW pre-allocation for GRPO.

        Key design:
        - Override async_infer() to pre-allocate DSWs and tasks BEFORE launching rollouts
        - At batch start: compute total_groups = len(requests) / num_generations
        - Randomly select unique DSWs, verify open ports, query task lists
        - Fix one task_id per DSW for the round, then parallel-sample across ports
        - Each env step is an INDEPENDENT single-turn inference
        """

        def __init__(self, infer_engine=None, max_turns: Optional[int] = None, **kwargs):
            super().__init__(infer_engine, max_turns, **kwargs)
            self.gym_env_name = kwargs.get("gym_env", "mcp")
            self.server_pool = MCPServerPool()

            # Optional history compression for earlier turns in final training messages.
            # Default OFF for strict backward compatibility.
            compress_flag = str(os.environ.get("MCP_COMPRESS_HISTORY", "0")).strip().lower()
            self.compress_history_enabled = compress_flag in {"1", "true", "yes", "on"}
            self.compress_keep_last_k = max(1, int(os.environ.get("MCP_COMPRESS_KEEP_LAST_K", "3")))
            self.compress_last_obs_max_chars = max(200, int(os.environ.get("MCP_COMPRESS_LAST_OBS_MAX_CHARS", "1200")))

            # Per-round allocation: set by async_infer(), consumed by run()
            # Maps prompt_id -> {dsw_idx, task_id, ports: List[str]}
            self._round_allocation = {}
            self._round_dsw_port_queues = {}
            self._bad_ports = set()
            self._tried_tasks = {}
            self._round_lock = None  # Lazy init

            # Concurrency limiter: cap simultaneous active MCP sessions to avoid
            # event loop starvation when many anyio task groups run concurrently.
            # Each active session spawns 2 anyio tasks (sse_reader + post_writer).
            # With 30+ concurrent sessions in the vLLM event loop, IO scheduling
            # latency grows and finish_with_id responses can be silently delayed.
            # Max 16 concurrent sessions is safe (tested: 30 work in isolation,
            # but the vLLM background loop competes for event loop time).
            _mcp_max_concurrent = int(os.environ.get("MCP_MAX_CONCURRENT", "24"))
            self._rollout_semaphore = asyncio.Semaphore(_mcp_max_concurrent)
            print(
                f"[MCPMultiTurnScheduler] MCP concurrency limit: {_mcp_max_concurrent} "
                f"(set MCP_MAX_CONCURRENT to override)"
            )

            print(f"[MCPMultiTurnScheduler] Initialized with max_turns={max_turns}, gym_env={self.gym_env_name}")
            print(
                f"[MCPMultiTurnScheduler] Server pool: {self.server_pool.num_dsw} DSW instances, "
                f"{self.server_pool.num_servers} total endpoints"
            )
            print(
                f"[MCPMultiTurnScheduler] History compression: enabled={self.compress_history_enabled}, "
                f"keep_last_k={self.compress_keep_last_k}, "
                f"last_obs_max_chars={self.compress_last_obs_max_chars}"
            )

        def _build_compressed_history_prompt(
            self,
            task: str,
            scene_description: str,
            latest_progress_step: str,
            last_action: Dict[str, Any],
            last_obs: str,
            ask_count: int,
            hand_state_description: str,
        ) -> str:
            """Build a compact user prompt for old trajectory turns."""
            clipped_obs = last_obs or ""
            if len(clipped_obs) > self.compress_last_obs_max_chars:
                clipped_obs = clipped_obs[: self.compress_last_obs_max_chars] + " ...[truncated]"

            return MCP_COMPRESSED_HISTORY_TEMPLATE.format(
                task=task,
                latest_progress_step=latest_progress_step,
                last_action=json.dumps(last_action, ensure_ascii=False),
                last_obs=clipped_obs,
                ask_count=ask_count,
                hand_state_description=hand_state_description,
            )

        async def _verify_dsw_ports(self, dsw_idx: int) -> List[str]:
            """
            Verify which ports on a DSW have a live MCP server via SSE handshake.

            Only checks liveness (fast, no MCP session created). A live SSE
            endpoint returns 'event: endpoint' as its first line. We read that
            one line then close — no MCP initialize or tool calls, so the
            server's single-task TaskManager is never touched.

            Returns list of alive server URLs for this DSW.
            """
            import httpx

            urls = self.server_pool.get_dsw_urls(dsw_idx)
            if not urls:
                return []

            SSE_TIMEOUT = 20.0

            async def probe(url):
                try:
                    async with httpx.AsyncClient() as client:
                        async with client.stream("GET", url, timeout=SSE_TIMEOUT) as resp:
                            async for _line in resp.aiter_lines():
                                return url
                except Exception:
                    return None

            results = await asyncio.gather(*[probe(u) for u in urls])
            working = [r for r in results if isinstance(r, str)]

            print(f"[MCPMultiTurnScheduler] DSW[{dsw_idx}] port check: {len(working)}/{len(urls)} alive")
            return working

        async def async_infer(
            self,
            infer_requests: List[Union["RolloutInferRequest", Dict[str, Any]]],
            request_config: "RequestConfig",
            *,
            use_tqdm: Optional[bool] = None,
            **kwargs,
        ) -> List["RolloutOutput"]:
            """
            Batch-aware rollout with pre-allocated DSWs and tasks.

            Flow:
            1. Group requests by prompt_id → determine total_groups
            2. Randomly select total_groups unique DSWs from pool
            3. Verify open ports on each selected DSW
            4. Query each DSW for task list, pick one random task_id per DSW
            5. Store allocation in self._round_allocation
            6. Launch all rollout tasks (they read from _round_allocation)
            7. After all done, release DSWs
            """
            from swift.llm.template import RolloutInferRequest as RIR
            import random as _random

            assert request_config.n == 1

            # Convert dict requests to RolloutInferRequest
            converted = []
            for req in infer_requests:
                if isinstance(req, dict):
                    converted.append(RIR(**req))
                else:
                    converted.append(req)

            # === Step 1: Group by message content ===
            # prompt_id does NOT survive the HTTP round-trip (ChatCompletionRequest
            # drops data_dict during serialization). So we compute groups ourselves
            # using the same approach as ms-swift's _add_prompt_id_to_inputs:
            # hash the messages content to identify which requests share the same prompt.
            import json as _json
            import hashlib as _hashlib

            prompt_groups = {}  # computed_pid -> [request_indices]
            messages_to_pid = {}  # messages_hash -> computed_pid
            pid_counter = 0

            for idx, req in enumerate(converted):
                # Hash the messages to identify the prompt group
                messages = req.messages if hasattr(req, "messages") else []
                msg_key = _json.dumps(messages, ensure_ascii=False, sort_keys=True)
                msg_hash = _hashlib.md5(msg_key.encode()).hexdigest()[:12]

                if msg_hash not in messages_to_pid:
                    messages_to_pid[msg_hash] = f"group_{pid_counter}"
                    pid_counter += 1

                pid = messages_to_pid[msg_hash]
                prompt_groups.setdefault(pid, []).append(idx)

                # Store computed pid in data_dict so _run_single_trajectory can use it
                if not hasattr(req, "data_dict") or req.data_dict is None:
                    req.data_dict = {}
                req.data_dict["prompt_id"] = pid

            total_groups = len(prompt_groups)
            total_requests = len(converted)
            num_generations = total_requests // total_groups if total_groups > 0 else 1

            print(
                f"[MCPMultiTurnScheduler] Message-hash grouping: {total_requests} requests -> "
                f"{total_groups} groups (num_generations={num_generations})"
            )
            for pid, indices in prompt_groups.items():
                print(f"  {pid}: {len(indices)} requests (indices {indices[0]}..{indices[-1]})")

            # === Step 0: Wait for startup connectivity probe + hard-halt if no usable DSWs ===
            # The probe runs once per process (in MCPServerPool._startup_probe) and marks
            # permanently-dead DSWs. Stop immediately when no live DSW remains instead
            # of spinning through a batch that cannot make progress.
            await self.server_pool.wait_startup_probe(timeout=120.0)
            dead_dsws = set(getattr(self.server_pool, "dead_dsws", set()))
            live_dsw_indices = [i for i in range(self.server_pool.num_dsw) if i not in dead_dsws]
            if not live_dsw_indices:
                raise RuntimeError(
                    "[MCPMultiTurnScheduler] ABORT: all DSWs failed the startup connectivity probe "
                    "(3 retries per DSW, every port unreachable). Check SSH tunnels / remote MCP servers."
                )
            print(f"\n{'=' * 60}")
            print(f"[MCPMultiTurnScheduler] === BATCH PRE-ALLOCATION ===")
            print(f"[MCPMultiTurnScheduler] Total requests: {total_requests}")
            print(f"[MCPMultiTurnScheduler] Total groups (unique tasks): {total_groups}")
            print(f"[MCPMultiTurnScheduler] num_generations per group: {num_generations}")
            print(
                f"[MCPMultiTurnScheduler] Available DSW instances: {self.server_pool.num_dsw} "
                f"(dead: {sorted(dead_dsws) if dead_dsws else 'none'})"
            )
            print(f"{'=' * 60}")

            # === Steps 2-4: Select DSWs, verify ports, fetch tasks — with fallback ===
            # We have 9 DSWs but only need `total_groups` (e.g. 3).
            # Try each candidate; if port check or task fetch fails, skip to next DSW.
            available_dsw_indices = [i for i in range(self.server_pool.num_dsw) if i not in dead_dsws]
            _random.shuffle(available_dsw_indices)

            # First, verify all DSWs in parallel to know which are actually alive
            all_port_checks = await asyncio.gather(*[self._verify_dsw_ports(di) for di in available_dsw_indices])
            dsw_open_ports_map = {}  # dsw_idx -> [working_urls]
            for di, ports in zip(available_dsw_indices, all_port_checks):
                dsw_open_ports_map[di] = ports
                if not ports:
                    print(f"[MCPMultiTurnScheduler] DSW[{di}] has 0 open ports, skipping")

            # Filter to DSWs with at least 1 open port
            alive_dsw = [di for di in available_dsw_indices if dsw_open_ports_map.get(di)]
            print(
                f"[MCPMultiTurnScheduler] Alive DSWs with open ports: {alive_dsw} ({len(alive_dsw)}/{len(available_dsw_indices)})",
                flush=True,
            )

            # === GUARD: handle transient probe failures (e.g. all SSE handshakes
            # timing out right after adapter weight sync). Without this, a single
            # bad probe round causes a ZeroDivisionError at the allocation step
            # below and brings down the trainer. Retry once with a fresh probe
            # before giving up. ===
            if not alive_dsw:
                print(
                    "[MCPMultiTurnScheduler] No alive DSWs on first probe — sleeping 5s and retrying once...",
                    flush=True,
                )
                await asyncio.sleep(5.0)
                retry_checks = await asyncio.gather(*[self._verify_dsw_ports(di) for di in available_dsw_indices])
                for di, ports in zip(available_dsw_indices, retry_checks):
                    dsw_open_ports_map[di] = ports
                alive_dsw = [di for di in available_dsw_indices if dsw_open_ports_map.get(di)]
                print(f"[MCPMultiTurnScheduler] After retry, alive DSWs: {alive_dsw}", flush=True)

            if not alive_dsw:
                raise RuntimeError(
                    f"[MCPMultiTurnScheduler] ABORT: 0 alive DSWs out of "
                    f"{len(available_dsw_indices)} available after retry. "
                    f"Likely cause: SSE probes all timed out under post-adapter-sync load, "
                    f"or all remote MCP servers / SSH tunnels unreachable. "
                    f"Check `ps -ef | grep 'ssh.*-L'` and curl each localhost:180XX/sse."
                )

            if len(alive_dsw) < total_groups:
                print(
                    f"[MCPMultiTurnScheduler] WARNING: Only {len(alive_dsw)} alive DSWs for {total_groups} groups!",
                    flush=True,
                )

            # Assign DSWs to groups with fallback: try fetch_task_list, skip on failure
            self._round_allocation = {}
            prompt_ids = list(prompt_groups.keys())
            used_dsw_set = set()

            for group_idx, pid in enumerate(prompt_ids):
                assigned = False
                # Try candidates until one works
                while not assigned:
                    # Pick next candidate DSW (prefer unused, then reuse)
                    di = None
                    for candidate in alive_dsw:
                        if candidate not in used_dsw_set:
                            di = candidate
                            break
                    if di is None:
                        # All DSWs used, reuse from alive pool
                        di = alive_dsw[group_idx % len(alive_dsw)]

                    # === DEADLOCK PREVENTION: DSW reuse must use the SAME task_id ===
                    # The server has a single-episode lock per port: once finish_with_id(task_X)
                    # is called, calling finish_with_id(task_Y != task_X) on the same port
                    # BLOCKS INDEFINITELY until task_X's episode completes. When a DSW is
                    # reused for a second group (due to fewer DSWs than groups), the second
                    # group picks up ports that already have the first group's task loaded.
                    # Assigning a different task would guarantee a deadlock. Force same task.
                    existing_alloc = next((a for a in self._round_allocation.values() if a["dsw_idx"] == di), None)
                    if existing_alloc is not None:
                        task_id = existing_alloc["task_id"]
                        ports = existing_alloc["ports"]
                        self._round_allocation[pid] = {
                            "dsw_idx": di,
                            "task_id": task_id,
                            "ports": ports,
                            "task_list": existing_alloc["task_list"],
                        }
                        group_size = len(prompt_groups[pid])
                        print(
                            f"[MCPMultiTurnScheduler] Group '{pid}' -> DSW[{di}] "
                            f"(REUSE, same task_id={task_id}, {len(ports)} ports, "
                            f"{group_size} rollouts will queue behind first group)"
                        )
                        assigned = True
                        continue

                    ports = dsw_open_ports_map[di]

                    # Probe EVERY verified port of this DSW and partition them into
                    # scenes by task-list signature. The DSW topology (URLs grouped by
                    # index) assumes all ports share one scene, but in practice some
                    # servers may have been restarted with different task sets. Using
                    # only the majority scene avoids allocating ports whose scene
                    # cannot serve the selected task. This prevents infinite
                    # "task not found" retry loops and zero-reward batches.
                    try:
                        per_port = await self.server_pool.fetch_task_lists_per_port(di, verified_urls=ports)
                    except Exception as e:
                        print(f"[MCPMultiTurnScheduler] DSW[{di}] per-port fetch failed: {e}, skipping")
                        alive_dsw = [x for x in alive_dsw if x != di]
                        if not alive_dsw:
                            raise RuntimeError("No alive DSWs remaining!")
                        continue

                    if not per_port:
                        print(f"[MCPMultiTurnScheduler] DSW[{di}] returned empty task list on every port, skipping")
                        alive_dsw = [x for x in alive_dsw if x != di]
                        if not alive_dsw:
                            raise RuntimeError("No alive DSWs remaining!")
                        continue

                    # Group ports by task-set signature (frozenset of task_ids).
                    scene_groups: Dict[frozenset, List[str]] = {}
                    for url, tl in per_port.items():
                        if not tl:
                            continue
                        sig = frozenset(tl)
                        scene_groups.setdefault(sig, []).append(url)

                    if not scene_groups:
                        print(f"[MCPMultiTurnScheduler] DSW[{di}] all ports returned empty task lists, skipping")
                        alive_dsw = [x for x in alive_dsw if x != di]
                        if not alive_dsw:
                            raise RuntimeError("No alive DSWs remaining!")
                        continue

                    # Pick the scene hosted by the most ports. Minority-scene ports
                    # are dropped from this round's allocation so that sampled tasks
                    # are guaranteed to load on every port the rollouts will use.
                    majority_sig, majority_urls = max(scene_groups.items(), key=lambda kv: len(kv[1]))
                    if len(scene_groups) > 1:
                        minority_count = sum(len(u) for sig, u in scene_groups.items() if sig != majority_sig)
                        print(
                            f"[MCPMultiTurnScheduler] DSW[{di}] scene heterogeneity detected: "
                            f"{len(scene_groups)} distinct task sets across {len(per_port)} ports. "
                            f"Using majority scene ({len(majority_urls)} ports, {len(majority_sig)} tasks); "
                            f"excluding {minority_count} mismatched ports."
                        )

                    tl = list(majority_sig)
                    ports = majority_urls

                    task_id = _random.choice(tl)
                    self._round_allocation[pid] = {
                        "dsw_idx": di,
                        "task_id": task_id,
                        "ports": ports,
                        "task_list": list(tl),  # cached — avoids live fetch_task_list during rollout
                    }
                    used_dsw_set.add(di)
                    group_size = len(prompt_groups[pid])
                    if len(ports) < group_size:
                        print(
                            f"[MCPMultiTurnScheduler] ⚠️ WARNING: Group '{pid}' has {group_size} rollouts "
                            f"but only {len(ports)} ports on DSW[{di}]. "
                            f"Rollouts will queue and wait for port release."
                        )
                    print(f"[MCPMultiTurnScheduler] Group '{pid}' -> DSW[{di}], task_id={task_id}, {len(ports)} ports")
                    assigned = True

            print(f"[MCPMultiTurnScheduler] Pre-allocation complete. Launching {total_requests} rollouts...")

            # === Step 5: Track port usage per group for round-robin ===
            # Each group has a port queue for its rollouts
            # === Global per-DSW port queues ===
            # All prompt groups sharing the same DSW draw from a single queue,
            # so a port taken by group A cannot be grabbed by group B.
            # The server's TaskManager is single-slot per port, so this
            # prevents cross-group request/response mixups.
            self._round_dsw_port_queues = {}  # dsw_idx -> asyncio.Queue[url]
            for pid, alloc in self._round_allocation.items():
                di = alloc["dsw_idx"]
                if di not in self._round_dsw_port_queues:
                    q = asyncio.Queue()
                    for url in alloc["ports"]:
                        q.put_nowait(url)
                    self._round_dsw_port_queues[di] = q

            # === GRPO Grouping Verification Log ===
            print(f"\n[MCPMultiTurnScheduler] === GRPO GROUPING VERIFICATION ===")
            for pid, indices in prompt_groups.items():
                alloc = self._round_allocation.get(pid, {})
                print(
                    f"  prompt_id={pid}: {len(indices)} rollouts -> DSW[{alloc.get('dsw_idx')}], "
                    f"task_id={alloc.get('task_id')}, {len(alloc.get('ports', []))} ports"
                )
            print(
                f"[MCPMultiTurnScheduler] All {total_groups} groups pre-allocated. "
                f"Each group gets the SAME task_id on the SAME DSW."
            )
            print(f"{'=' * 60}\n")

            # === Step 6: Launch all rollout tasks ===
            async def _infer_async_single(infer_request, request_config, **kwargs):
                return await self.run(infer_request, request_config, **kwargs)

            tasks = [_infer_async_single(req, request_config, **kwargs) for req in converted]
            if use_tqdm is None:
                use_tqdm = len(converted) > 1

            # Watchdog: log semaphore/queue state every 60s so silent hangs are visible.
            _batch_start = time.time()
            _batch_done = False
            _mcp_limit = self._rollout_semaphore._value  # snapshot of initial limit

            async def _watchdog():
                while not _batch_done:
                    await asyncio.sleep(60)
                    if _batch_done:
                        break
                    elapsed = time.time() - _batch_start
                    free_slots = self._rollout_semaphore._value
                    active_sessions = _mcp_limit - free_slots
                    dsw_queues = {di: q.qsize() for di, q in self._round_dsw_port_queues.items()}
                    print(
                        f"\n[WATCHDOG] Batch running {elapsed:.0f}s: "
                        f"active MCP sessions={active_sessions}/{_mcp_limit}, "
                        f"DSW port queues (free ports)={dsw_queues}"
                    )

            watchdog_task = asyncio.create_task(_watchdog())
            try:
                results = await self.infer_engine._batch_infer_stream(tasks, request_config.stream, use_tqdm, None)
            finally:
                _batch_done = True
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass

            # Flatten results
            flattened_results = []
            for result in results:
                if isinstance(result, list):
                    flattened_results.extend(result)
                else:
                    flattened_results.append(result)

            # === Step 7: Cleanup round allocation ===
            self._round_allocation = {}
            self._round_dsw_port_queues = {}
            self._bad_ports = set()
            print(f"[MCPMultiTurnScheduler] Round complete. Released all pre-allocations.")

            return flattened_results

        async def run(
            self, infer_request: "RolloutInferRequest", request_config: "RequestConfig", **kwargs
        ) -> "RolloutOutput":
            """
            Execute MCP environment rollout using pre-allocated DSW and task_id.

            Reads from self._round_allocation (set by async_infer) to get:
            - dsw_idx: which DSW to use
            - task_id: which task to load
            - port: acquired from round-robin port queue

            Retry logic: retry entire trajectory on failure.
            """
            from swift.llm.infer.protocol import RolloutOutput

            trajectory_id = infer_request.uuid
            env_config = infer_request.data_dict.get("env_config", {})
            prompt_id = infer_request.data_dict.get("prompt_id")

            max_trajectory_retries = 2
            TRAJECTORY_TIMEOUT = 900.0  # 15 minutes per attempt (Isaac restart adds ~210s)

            for trajectory_attempt in range(max_trajectory_retries):
                try:
                    return await asyncio.wait_for(
                        self._run_single_trajectory(
                            infer_request, request_config, env_config, trajectory_id, trajectory_attempt, **kwargs
                        ),
                        timeout=TRAJECTORY_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    print(
                        f"\n[MCPMultiTurnScheduler] ⚠️ TRAJECTORY TIMEOUT after {TRAJECTORY_TIMEOUT}s! "
                        f"(attempt {trajectory_attempt + 1})"
                    )
                    print(f"  Will retry with a different port/task...")
                    continue
                except Exception as e:
                    error_msg = str(e)
                    print(
                        f"\n[MCPMultiTurnScheduler] Trajectory attempt {trajectory_attempt + 1} failed: {error_msg[:200]}"
                    )
                    # DSW-level failures (exhausted tasks) won't recover with retries
                    if "exhausted" in error_msg.lower():
                        print(f"[MCPMultiTurnScheduler] DSW exhausted, skipping remaining retries")
                        break
                    if trajectory_attempt < max_trajectory_retries - 1:
                        print(f"  Will retry... ({max_trajectory_retries - trajectory_attempt - 1} attempts remaining)")
                        await asyncio.sleep(1.0)
                    continue

            print(f"\n[MCPMultiTurnScheduler] All {max_trajectory_retries} attempts failed!")
            return self._create_infrastructure_failure_output(infer_request, max_trajectory_retries)

        async def _run_single_trajectory(
            self, infer_request, request_config, env_config, trajectory_id: str, trajectory_attempt: int, **kwargs
        ):
            """Execute a single trajectory attempt using pre-allocated DSW and task_id."""
            from swift.llm.infer.protocol import RolloutOutput

            server_url = None
            dsw_idx = None
            env = None
            prompt_id = infer_request.data_dict.get("prompt_id")
            tid = trajectory_id[:8]
            _semaphore_acquired = False
            trajectory_completed = False  # set True only when we return RolloutOutput

            try:
                # === STEP 1: Get pre-allocated DSW + task_id, acquire port from round queue ===
                if prompt_id and prompt_id in self._round_allocation:
                    alloc = self._round_allocation[prompt_id]
                    dsw_idx = alloc["dsw_idx"]
                    task_id = alloc["task_id"]

                    # Acquire a port from the queue — MUST wait until one is returned.
                    # Acquire a port from the DSW's global queue.
                    # All prompt groups sharing this DSW draw from the same queue,
                    # so a port taken by one group cannot be used by another.
                    port_queue = self._round_dsw_port_queues.get(dsw_idx)
                    if port_queue is None:
                        raise RuntimeError(f"No port queue for DSW[{dsw_idx}]")
                    try:
                        server_url = await asyncio.wait_for(port_queue.get(), timeout=3600.0)
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"Port acquire timeout for prompt_id={prompt_id} on DSW[{dsw_idx}]: "
                            f"all ports busy for >3600s (queue size 0, "
                            f"allocated {len(alloc.get('ports', []))} ports)"
                        )

                    print(f"\n[{tid}] Starting rollout (pre-allocated)...")
                    print(
                        f"[{tid}] DSW[{dsw_idx}], task_id={task_id}, server={server_url} (attempt {trajectory_attempt + 1})"
                    )
                elif prompt_id:
                    # prompt_id exists but not in pre-allocation (shouldn't happen normally)
                    server_url = await self.server_pool.acquire(timeout=3600.0)
                    task_id = None
                    print(f"\n[{tid}] Starting rollout (fallback, prompt_id={prompt_id} not pre-allocated)...")
                    print(f"[{tid}] Server URL: {server_url} (attempt {trajectory_attempt + 1})")
                else:
                    # Fallback: no prompt_id (shouldn't happen in GRPO, but safety net)
                    server_url = await self.server_pool.acquire(timeout=3600.0)
                    task_id = None
                    print(f"\n[{tid}] Starting rollout (compat mode, no prompt_id)...")
                    print(f"[{tid}] Server URL: {server_url} (attempt {trajectory_attempt + 1})")

                # Acquire concurrency semaphore BEFORE opening SSE connection.
                # Limits simultaneous anyio task groups (sse_reader + post_writer per
                # session) to prevent event loop starvation in the vLLM async context.
                await self._rollout_semaphore.acquire()
                _semaphore_acquired = True
                print(f"[{tid}] Semaphore acquired (connecting to MCP...)")

                # Create environment with persistent connection
                env = MCPSingleEnv(env_id=0, server_url=server_url, env_config=env_config)
                await env.connect()

                # Reset environment — use finish_with_id if we have a cached task_id
                MAX_RESET_RETRIES = 5
                reset_success = False
                observation, info = None, {}

                for reset_attempt in range(MAX_RESET_RETRIES):
                    # NOTE: Do NOT sync task_id from peer rollouts here.
                    # The server has a single-episode lock: once finish_with_id(task_X) is
                    # called on a port, calling finish_with_id(task_Y != task_X) on the same
                    # port blocks indefinitely until task_X is completed. Changing task_id
                    # mid-rollout while on the same port guarantees a hang.

                    # On retries (stale SSE or transient error), disconnect and reconnect
                    # to flush the SSE stream state. finish_with_id(same_task) on a
                    # reconnected session works fine (confirmed: ~3.7s in experiments).
                    if reset_attempt > 0 and env is not None:
                        try:
                            await asyncio.wait_for(env.disconnect(), timeout=4.0)
                        except BaseException:
                            pass
                        try:
                            await asyncio.wait_for(env.connect(), timeout=15.0)
                        except (asyncio.TimeoutError, Exception) as reconn_err:
                            print(f"[{tid}] Reconnect failed/timeout: {reconn_err}, skipping remaining resets")
                            break

                    print(f"[{tid}] Reset attempt {reset_attempt + 1}/{MAX_RESET_RETRIES} (task_id={task_id})...")
                    try:
                        observation, info = await asyncio.wait_for(env.reset(task_id=task_id), timeout=45.0)
                    except asyncio.TimeoutError:
                        print(
                            f"[{tid}] env.reset() timed out after 45s (server may be blocked). "
                            f"Skipping remaining resets."
                        )
                        break

                    # Check for reset errors
                    if info.get("error"):
                        error_msg = f"Environment reset failed: {observation}"
                        print(f"[{tid}] Reset ERROR: {error_msg}")

                        # "Task not found" means this port runs a different scene.
                        # Swap to another port from the queue and retry.
                        if task_id and should_treat_reset_error_as_task_not_found(observation, info):
                            print(
                                f"[{tid}] ⚠️ Port {server_url} has wrong scene (task '{task_id}' not found). Swapping port..."
                            )
                            bad_port = server_url
                            # Try to get a different port from the DSW's global queue
                            port_queue = self._round_dsw_port_queues.get(dsw_idx)
                            if port_queue:
                                # Return bad port tagged so we don't pick it again
                                if not isinstance(getattr(self, "_bad_ports", None), set):
                                    self._bad_ports = set()
                                self._bad_ports.add(bad_port)
                                # Put bad port back (others may need it for a different scene)
                                port_queue.put_nowait(bad_port)
                                # Try to get a good port
                                found_good = False
                                for _swap in range(port_queue.qsize()):
                                    try:
                                        candidate = await asyncio.wait_for(port_queue.get(), timeout=5.0)
                                    except asyncio.TimeoutError:
                                        break
                                    if candidate not in self._bad_ports:
                                        server_url = candidate
                                        await env.disconnect()  # Close old connection
                                        env = MCPSingleEnv(env_id=0, server_url=server_url, env_config=env_config)
                                        await env.connect()
                                        print(f"[{tid}] Swapped to port {server_url}")
                                        found_good = True
                                        break
                                    else:
                                        port_queue.put_nowait(candidate)
                                if not found_good:
                                    # bad_port was already returned to the queue above (line: put bad_port).
                                    # Clear server_url so the finally block does NOT double-release it.
                                    server_url = None
                                    raise RuntimeError(
                                        f"All ports on DSW[{dsw_idx}] have wrong scene for task '{task_id}'"
                                    )
                            continue

                        # Check if this is a recoverable "Error spawning episode" error
                        if "error spawning episode" in str(observation).lower():
                            if reset_attempt < MAX_RESET_RETRIES - 1:
                                await asyncio.sleep(2.0)
                                continue

                        # Stale SSE (task_id mismatch): the server actually loaded the right
                        # task but we read a stale response. Reconnect-on-retry (above) will
                        # flush the SSE state; retry with the SAME task_id will succeed.
                        if info.get("stale_sse"):
                            print(f"[{tid}] Stale SSE detected, will reconnect and retry same task_id={task_id}")
                            continue

                        # Genuinely invalid task (target_diffs=0, empty world_graph, etc.):
                        # The server has this task loaded. We CANNOT call finish_with_id with a
                        # different task_id on the same port — it would block indefinitely.
                        # Swap to a different port from the DSW queue (fresh port has no episode
                        # loaded, so finish_with_id(new_task) will work).
                        if info.get("invalid_task") and prompt_id and prompt_id in self._round_allocation:
                            if not hasattr(self, "_tried_tasks") or not isinstance(self._tried_tasks, dict):
                                self._tried_tasks = {}
                            tried = self._tried_tasks.setdefault(dsw_idx, set())
                            tried.add(task_id)

                            alloc = self._round_allocation[prompt_id]
                            old_task_id = alloc.get("task_id")
                            try:
                                tl = alloc.get("task_list") or []
                                if not tl:
                                    print(f"[{tid}] ⚠️ task_list cache miss, falling back to live fetch")
                                    verified = alloc.get("ports", [])
                                    raw_tl = await self.server_pool.fetch_task_list(dsw_idx, verified_urls=verified)
                                    tl = (
                                        self.server_pool._normalize_task_list(raw_tl)
                                        if not isinstance(raw_tl, list)
                                        else raw_tl
                                    )
                                candidates = [t for t in tl if t not in tried]
                                if not candidates:
                                    print(f"[{tid}] All {len(tried)} task_ids exhausted on DSW[{dsw_idx}], giving up")
                                    raise RuntimeError(
                                        f"DSW[{dsw_idx}] exhausted: tried {len(tried)} tasks, none valid"
                                    )
                                new_task_id = random.choice(candidates)
                                self._round_allocation[prompt_id]["task_id"] = new_task_id
                                task_id = new_task_id
                                print(f"[{tid}] ⚠️ Invalid task '{old_task_id}' (genuine), resampled -> '{new_task_id}'")

                                # CRITICAL: swap port — current port has old task loaded,
                                # calling finish_with_id(new_task) on it would block.
                                # EXCEPTION: if the task was "not found", the server returned
                                # immediately (no episode started) → port IS free, no swap needed.
                                _task_not_found = should_treat_reset_error_as_task_not_found(observation, info)
                                port_queue = self._round_dsw_port_queues.get(dsw_idx)
                                if port_queue and port_queue.qsize() > 0:
                                    stuck_port = server_url
                                    try:
                                        new_server_url = await asyncio.wait_for(port_queue.get(), timeout=5.0)
                                    except asyncio.TimeoutError:
                                        new_server_url = None
                                    if new_server_url:
                                        # Return the stuck port to the queue — it'll recover
                                        # when its current episode eventually completes.
                                        port_queue.put_nowait(stuck_port)
                                        try:
                                            await asyncio.wait_for(env.disconnect(), timeout=4.0)
                                        except BaseException:
                                            pass
                                        server_url = new_server_url
                                        env = MCPSingleEnv(env_id=0, server_url=server_url, env_config=env_config)
                                        try:
                                            await asyncio.wait_for(env.connect(), timeout=15.0)
                                        except (asyncio.TimeoutError, Exception) as ce:
                                            print(f"[{tid}] Connect to new port failed: {ce}")
                                            raise RuntimeError(f"Could not connect to new port: {ce}")
                                        print(f"[{tid}] Swapped port {stuck_port} -> {server_url} for new task")
                                    elif _task_not_found:
                                        # Port is free (task was never loaded), reuse same port.
                                        print(
                                            f"[{tid}] Task not found → port free, retrying '{new_task_id}' on same port"
                                        )
                                    else:
                                        print(f"[{tid}] No spare port available for new task, giving up")
                                        raise RuntimeError("No spare port for task swap")
                                elif _task_not_found:
                                    # Port is free (task was never loaded), reuse same port.
                                    print(f"[{tid}] Task not found → port free, retrying '{new_task_id}' on same port")
                                else:
                                    print(f"[{tid}] No spare port available for task swap, giving up")
                                    raise RuntimeError("No spare port for task swap")
                                continue  # retry reset with new task_id on new port
                            except RuntimeError:
                                raise
                            except Exception as swap_err:
                                print(f"[{tid}] Failed to resample/swap task: {swap_err}")
                                # fall through to raise

                        raise RuntimeError(error_msg)

                    # Check for "done" (all episodes completed)
                    if info.get("done"):
                        print(f"[{tid}] All episodes completed")
                        raise RuntimeError("All episodes completed - no more tasks available")

                    # Validate that we got a proper task instruction
                    task = info.get("instruction", observation)

                    # Check for "Error spawning episode" in task - this is recoverable
                    if task and "error spawning episode" in task.lower():
                        print(f"[{tid}] MCP spawning error, retrying...")
                        if reset_attempt < MAX_RESET_RETRIES - 1:
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            raise RuntimeError(f"Reset failed after {MAX_RESET_RETRIES} attempts: {task[:200]}")

                    # Check for other invalid tasks (non-recoverable)
                    if not task or "connection" in task.lower():
                        error_msg = f"Invalid task received: {task if task else 'empty'}"
                        print(f"[{tid}] Reset ERROR: {error_msg}")
                        raise RuntimeError(error_msg)

                    # Check for "All evaluation episodes completed" - this means the server ran out of tasks
                    if "all evaluation episodes completed" in task.lower():
                        error_msg = f"MCP server exhausted: {task}"
                        print(f"[{tid}] Reset ERROR: {error_msg}")
                        raise RuntimeError(error_msg)

                    # Success!
                    reset_success = True
                    break

                if not reset_success:
                    raise RuntimeError(f"Reset failed after {MAX_RESET_RETRIES} attempts")

                print(f"[{tid}] Task: {task[:80]}...")
                print(f"[{tid}] Target diffs: {info.get('target_diffs', 'N/A')}")

                scene_desc = info.get("scene_description", "")

                def _get_hand_state_description() -> str:
                    if env.hand_occupied:
                        inv_name = "an object"
                        if env.current_inv:
                            uid_part = str(env.current_inv)
                            if "_on_" in uid_part and "_at_" in uid_part:
                                uid_part = uid_part.split("_on_")[0]
                            inv_name = env.uid_to_category.get(uid_part, env.current_inv)
                        return f"OCCUPIED (holding: {inv_name})"
                    return "EMPTY (not holding any object)"

                # Tracking variables
                current_step = 0
                done = False
                total_reward = 0.0
                step_rewards = []
                trajectory_info = [info]

                # Ask tool limit: max_ask_limit = ceil(max_turns * 0.15)
                import math

                max_turns_val = self.max_turns or 20  # default to 20 if not set
                max_ask_limit = math.ceil(max_turns_val * 0.15)
                ask_count = 0

                # For prompt template - use HistoryManager for deterministic history
                # This aligns with qwen_client.py's approach
                history_manager = HistoryManager()
                last_action = {}
                last_obs = ""

                # For VLM: track images from previous step
                # Images are passed as base64 in VLM content format
                current_images = []  # List of {'data': base64, 'mimeType': str}
                current_grounded_graph = ""  # For compressed history ONLY (prevent cheating)

                # For final RolloutOutput - proper multi-turn format (user, assistant, user, assistant, ...)
                # This is CRITICAL: swift template expects alternating user/assistant messages
                # Do NOT create consecutive assistant messages - they will cause TypeError during merge
                all_user_prompts = []  # Store user prompts for proper interleaving (VLM content format)
                all_compressed_user_prompts = []  # Store compressed prompt variants for old turns
                all_completions = []  # Store assistant completions
                response = None

                print(f"[{tid}] Starting inference loop (max_turns={self.max_turns})...")

                # === STEP 2: Single-turn inference loop ===
                while not done and current_step < (self.max_turns or float("inf")):
                    print(f"\n[{tid}] === Step {current_step + 1} ===")

                    # Build fresh single-turn prompt (no history accumulation!)
                    # Use HistoryManager for deterministic task progress (aligns with qwen_client.py)
                    task_progress_str = history_manager.get_task_progress_str()
                    latest_progress_step = (
                        history_manager.accumulated_history[-1] if history_manager.accumulated_history else "{}"
                    )
                    user_text = MCP_PROMPT_TEMPLATE.format(
                        SCENE_DESCRIPTION=scene_desc,
                        TASK=task,
                        TASK_PROGRESS=task_progress_str,
                        LAST_ACTION=json.dumps(last_action),
                        LAST_OBS=last_obs,
                        max_ask_limit=max_ask_limit,
                        ask_count=ask_count,
                        hand_state_description=_get_hand_state_description(),
                    )

                    # Only append grounded info to compressed history, to NOT cheat the current real step
                    compressed_last_obs = last_obs
                    if current_grounded_graph:
                        compressed_last_obs += f"\n[Grounded Visible Context]: {current_grounded_graph}"

                    compressed_user_text = self._build_compressed_history_prompt(
                        task=task,
                        scene_description=scene_desc,
                        latest_progress_step=latest_progress_step,
                        last_action=last_action,
                        last_obs=compressed_last_obs,
                        ask_count=ask_count,
                        hand_state_description=_get_hand_state_description(),
                    )

                    # Build VLM content format: [{"type": "text", ...}, {"type": "image", ...}, ...]
                    # This is required for Qwen3-VL to properly process multiple images
                    user_content = [{"type": "text", "text": user_text}]

                    # Add images from previous step (if any)
                    for img_data in current_images:
                        # Use base64 data URL format for vLLM
                        mime_type = img_data.get("mimeType", "image/png")
                        base64_data = img_data.get("data", "")
                        image_url = f"data:{mime_type};base64,{base64_data}"
                        user_content.append({"type": "image", "image": image_url})

                    print(f"[{tid}] VLM: {len(current_images)} images")

                    # Single user message only - no conversation history!
                    # Use VLM content format for multimodal input
                    single_turn_messages = [{"role": "user", "content": user_content}]

                    current_request = deepcopy(infer_request)
                    current_request.messages = single_turn_messages

                    print(f"[{tid}] Last action: {last_action}")

                    # Store user prompt for final messages
                    # Note: For training, we only store text (images are for inference only)
                    all_user_prompts.append(user_text)
                    all_compressed_user_prompts.append(compressed_user_text)

                    # Get model response (single-turn inference)
                    response = await self.infer_engine.infer_async(current_request, request_config, **kwargs)
                    response_choice = response.choices[0]
                    completion = response_choice.message.content

                    # Guard against empty completion (model generated EOS immediately)
                    if not completion or not completion.strip():
                        print(
                            f"[{tid}] ⚠️ Empty completion from model (finish_reason={response_choice.finish_reason}), aborting trajectory"
                        )
                        break

                    all_completions.append(completion)

                    # Execute environment step
                    next_obs, reward, done, step_info = await env.step(completion)
                    try:
                        reward = float(reward)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(f"MCP environment returned a non-numeric reward: {reward!r}") from exc
                    if not math.isfinite(reward):
                        raise RuntimeError(f"MCP environment returned a non-finite reward: {reward!r}")
                    # Store observation text in step_info for trajectory log
                    step_info["observation"] = next_obs if next_obs else ""

                    # Get tool info for logging
                    tool_name = step_info.get("tool_name", "unknown")
                    tool_args = step_info.get("tool_args", {})
                    print(f"[{tid}] Action: {tool_name} | Reward: {reward:.3f} | Done: {done}")

                    # Check for connection errors in step - if so, we need to restart trajectory
                    if step_info.get("error") and "connection" in str(step_info.get("error", "")).lower():
                        raise RuntimeError(f"Connection error during step: {step_info.get('error')}")

                    # Update tracking
                    total_reward += reward
                    step_rewards.append(reward)
                    trajectory_info.append(step_info)

                    # Track ask tool usage - with debug output
                    if tool_name == "ask":
                        ask_count += 1
                        ask_question = tool_args.get("question", "")
                        ask_response = step_info.get("ask_response", next_obs)
                        print(f"[{tid}] [ASK] Q: {ask_question[:150]}")
                        print(f"[{tid}] [ASK] Response: {ask_response[:300]}")

                    # Check finish conditions
                    if done:
                        break

                    if response_choice.finish_reason == "length":
                        print(f"[{tid}] Response hit length limit")
                        break

                    # === Update state for next prompt using HistoryManager ===
                    # Extract new step from model's output and add to deterministic history
                    try:
                        if "</think>" in completion:
                            action_part = completion.split("</think>")[-1].strip()
                        else:
                            action_part = completion

                        parsed = json.loads(action_part)

                        # Extract History from model output and get the new step
                        if "summary" in parsed and "History" in parsed["summary"]:
                            gpt_history = parsed["summary"]["History"]
                            new_step = history_manager.extract_new_step_from_gpt_output(gpt_history)
                            if new_step:
                                history_manager.add_step(new_step)

                        # Update last_action for next prompt
                        if "next action" in parsed:
                            last_action = parsed["next action"]
                    except Exception as e:
                        # If parsing fails, add a generic step description
                        history_manager.add_step(f"Executed action (parse failed)")
                        last_action = {"raw": completion[:100]}

                    # Update last observation for next prompt (aligned with SFT format)
                    # Images are stored in step_info['images'] as base64 data
                    current_images = step_info.get("images", [])
                    if current_images:
                        last_obs = (
                            f"After performing the last action, you observed the image <image> and the text: {next_obs}"
                        )
                    else:
                        last_obs = f"After performing the last action, you observed the text: {next_obs}"

                    # Keep grounded world graph logic for compressed history ONLY, to NOT cheat the current step
                    current_grounded_graph = step_info.get("grounded_world_graph", "")

                    current_step += 1

                # === CRITICAL: Apply incompletion penalty if max_turns reached without completion ===
                # This prevents reward hacking: model cannot get 0 reward by doing nothing
                if not done and current_step >= (self.max_turns or float("inf")):
                    incompletion_penalty = -1.0
                    total_reward += incompletion_penalty
                    step_rewards.append(incompletion_penalty)  # Record as final "step"
                    print(f"[{tid}] PENALTY: Task not completed, -1.0 applied")

                # =====================================================
                # Build final messages for RolloutOutput
                # CRITICAL: Must be proper multi-turn format (user, assistant, user, assistant, ...)
                # swift template's _swift_prepare_inputs will try to merge consecutive messages
                # of the same role, and fails if content types differ (str vs list)
                # =====================================================
                final_messages = []

                # Interleave user prompts and completions to create proper conversation
                assert len(all_user_prompts) == len(all_completions), (
                    f"Mismatch: {len(all_user_prompts)} prompts vs {len(all_completions)} completions"
                )
                assert len(all_compressed_user_prompts) == len(all_user_prompts), (
                    f"Mismatch: {len(all_compressed_user_prompts)} compressed prompts vs {len(all_user_prompts)} prompts"
                )

                total_turns = len(all_user_prompts)
                compress_prefix_turns = max(0, total_turns - self.compress_keep_last_k)

                if self.compress_history_enabled and compress_prefix_turns > 0:
                    print(
                        f"[{tid}] Compressing old user prompts: {compress_prefix_turns}/{total_turns} turns "
                        f"(keep last {self.compress_keep_last_k} full)"
                    )

                for turn_idx, (user_prompt, compressed_user_prompt, completion) in enumerate(
                    zip(all_user_prompts, all_compressed_user_prompts, all_completions)
                ):
                    if self.compress_history_enabled and turn_idx < compress_prefix_turns:
                        selected_user_prompt = compressed_user_prompt
                    else:
                        selected_user_prompt = user_prompt

                    # User message - ensure string content
                    if isinstance(selected_user_prompt, list):
                        # VLM list format - extract text parts
                        text_parts = [
                            item.get("text", "")
                            for item in selected_user_prompt
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        selected_user_prompt = " ".join(text_parts)
                    final_messages.append({"role": "user", "content": str(selected_user_prompt)})

                    # Assistant message - ensure string content
                    if isinstance(completion, list):
                        # VLM list format - extract text parts
                        text_parts = [
                            item.get("text", "")
                            for item in completion
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        completion = " ".join(text_parts)
                    final_messages.append({"role": "assistant", "content": str(completion)})

                # Verify message structure (silent unless error)
                for i, msg in enumerate(final_messages):
                    content = msg.get("content", "")
                    if not isinstance(content, str):
                        print(f"[{tid}] WARNING: Non-string content at msg {i}, converting...")
                        msg["content"] = str(content)

                # === Hard truncate final_messages to fit vllm_max_model_len ===
                # Estimate tokens via char count (conservative: 3 chars/token).
                # Drop oldest user+assistant pairs until under the limit.
                # Always keep at least the last 2 messages (1 pair).
                _MAX_CHARS = int(os.environ.get("MCP_MAX_CONTEXT_CHARS", str(6144 * 3)))
                _total_chars = sum(len(m.get("content", "")) for m in final_messages)
                if _total_chars > _MAX_CHARS and len(final_messages) > 2:
                    _orig_turns = len(final_messages) // 2
                    while len(final_messages) > 2:
                        _total_chars = sum(len(m.get("content", "")) for m in final_messages)
                        if _total_chars <= _MAX_CHARS:
                            break
                        final_messages = final_messages[2:]  # drop oldest user+assistant pair
                    remaining_turns = len(final_messages) // 2
                    print(
                        f"[{tid}] Context truncated: {_orig_turns} -> {remaining_turns} turns "
                        f"(estimated chars {_total_chars} > limit {_MAX_CHARS})"
                    )
                    # If more than half the trajectory was dropped, zeroing reward is cleaner than
                    # training on a fragment where reward attribution is meaningless.
                    if remaining_turns < _orig_turns / 2:
                        print(
                            f"[{tid}] WARN: truncation dropped >{_orig_turns // 2} turns "
                            f"({_orig_turns} -> {remaining_turns}), zeroing reward to skip gradient update"
                        )
                        total_reward = 0.0

                # === CRITICAL: Log final reward being sent to trainer ===
                print(f"\n{'=' * 60}")
                print(f"[{tid}] TRAJECTORY COMPLETE")
                print(f"[{tid}] Total Steps: {current_step + 1}")
                print(f"[{tid}] Step Rewards: {step_rewards}")
                print(f"[{tid}] >>> TOTAL REWARD: {total_reward:.4f} <<<")
                print(f"[{tid}] Done: {done}")
                print(f"{'=' * 60}\n")
                sys.stdout.flush()  # Force immediate output

                # === Save structured trajectory log to JSON ===
                try:
                    from datetime import datetime

                    # Create trajectory log directory
                    log_dir = _runtime_path("MCP_TRAJECTORY_LOG_DIR", "output", "trajectory_logs")
                    os.makedirs(log_dir, exist_ok=True)

                    # Build structured trajectory data
                    trajectory_steps = []
                    for step_idx, (completion, step_inf) in enumerate(
                        zip(all_completions, trajectory_info[1:])
                    ):  # Skip initial info
                        # Extract action from completion
                        action_data = {}
                        try:
                            if "</think>" in completion:
                                action_part = completion.split("</think>")[-1].strip()
                            else:
                                action_part = completion
                            parsed = json.loads(action_part)
                            action_data = parsed.get("next action", {})
                        except:
                            action_data = {"raw": completion[:200]}

                        # Get observation (without images)
                        obs_text = step_inf.get("observation", "")
                        if not obs_text and step_idx + 1 < len(all_user_prompts):
                            # Extract from next prompt's LAST_OBS
                            next_prompt = all_user_prompts[step_idx + 1] if step_idx + 1 < len(all_user_prompts) else ""
                            if "After performing the last action, you observed the text:" in next_prompt:
                                obs_text = next_prompt.split(
                                    "After performing the last action, you observed the text:"
                                )[-1][:500]
                            elif "After performing the last action, you observed the image" in next_prompt:
                                obs_text = "<image>"

                        step_data = {
                            "step": step_idx,
                            "action": action_data,
                            "tool_name": step_inf.get("tool_name", ""),
                            "tool_args": step_inf.get("tool_args", {}),
                            "reward": float(step_rewards[step_idx]) if step_idx < len(step_rewards) else 0,
                            "observation": obs_text if obs_text else "",
                            "server_url": step_inf.get("server_url", ""),
                        }

                        # Per-step debug fields
                        if "reward_breakdown" in step_inf:
                            step_data["reward_breakdown"] = step_inf["reward_breakdown"]
                        if "tool_error" in step_inf:
                            step_data["tool_error"] = step_inf["tool_error"]
                        if "hand_violation" in step_inf:
                            step_data["hand_violation"] = step_inf["hand_violation"]
                        if "completion_rate" in step_inf:
                            step_data["completion_rate"] = round(float(step_inf["completion_rate"]), 4)
                        # Hand state AFTER this step executed
                        if "hand_occupied" in step_inf:
                            step_data["hand_occupied"] = step_inf["hand_occupied"]
                        # World graph snapshot after this step (for verifying diff calculations)
                        if "world_graph_snapshot" in step_inf:
                            step_data["world_graph"] = step_inf["world_graph_snapshot"]
                        # Error/invalid action flags
                        if step_inf.get("error"):
                            step_data["error"] = str(step_inf["error"])[:300]
                        if step_inf.get("invalid_action"):
                            step_data["invalid_action"] = True
                        # Timeout flag
                        if step_inf.get("timeout"):
                            step_data["timeout"] = True
                        # Server-side debug state (landmark, inv, pos, marker_map, world_graph)
                        if "server_debug" in step_inf:
                            step_data["server_debug"] = step_inf["server_debug"]
                        # MCP response metadata
                        if "mcp_num_content" in step_inf:
                            step_data["mcp_num_content"] = step_inf["mcp_num_content"]
                            step_data["mcp_response_types"] = step_inf.get("mcp_response_types", [])
                        # Raw VLM completion (truncated)
                        if "raw_completion" in step_inf:
                            step_data["raw_completion"] = step_inf["raw_completion"][:500]

                        # Add ask-specific debug info
                        if step_inf.get("tool_name") == "ask":
                            step_data["ask_question"] = step_inf.get("tool_args", {}).get("question", "")
                            ask_resp = step_inf.get("ask_response", obs_text)
                            step_data["ask_response"] = ask_resp[:1000] if ask_resp else ""
                            print(f"[{tid}] [ASK DEBUG] Q: {step_data['ask_question'][:100]}")
                            print(f"[{tid}] [ASK DEBUG] A: {step_data['ask_response'][:200]}")

                        trajectory_steps.append(step_data)

                    # === Build trajectory-level summary statistics ===
                    from collections import Counter

                    tool_names_seq = [s.get("tool_name", "?") for s in trajectory_steps]
                    tool_counter = Counter(tool_names_seq)
                    error_steps = [s["step"] for s in trajectory_steps if s.get("tool_error") or s.get("error")]
                    hand_violation_steps = [s["step"] for s in trajectory_steps if s.get("hand_violation")]

                    # Determine failure reason (use completion_rate, not reward threshold)
                    final_completion = (
                        trajectory_steps[-1].get("reward_breakdown", {}).get("completion_rate", 0)
                        if trajectory_steps
                        else 0
                    )
                    if done and final_completion == 1.0:
                        outcome = "success"
                    elif not done and current_step >= (self.max_turns or float("inf")):
                        outcome = "timeout"
                    elif done and total_reward <= 0:
                        outcome = "failed"
                    else:
                        outcome = "partial"

                    trajectory_log = {
                        "trajectory_id": trajectory_id,
                        "timestamp": datetime.now().isoformat(),
                        "task": task[:500],
                        "target_diffs": info.get("target_diffs", []),
                        "initial_world_graph": info.get("world_graph", {}),
                        "target_world_graph": info.get("target_world_graph", {}),
                        "total_reward": float(total_reward),
                        "success": done
                        and any(s.get("reward_breakdown", {}).get("completion_rate") == 1.0 for s in trajectory_steps),
                        "outcome": outcome,
                        "num_steps": len(trajectory_steps),
                        "done": done,
                        # Summary statistics for quick analysis
                        "summary": {
                            "action_sequence": " → ".join(tool_names_seq),
                            "tool_usage": dict(tool_counter),
                            "error_steps": error_steps,
                            "hand_violation_steps": hand_violation_steps,
                            "final_completion_rate": trajectory_steps[-1].get("completion_rate", None)
                            if trajectory_steps
                            else None,
                        },
                        "steps": trajectory_steps,
                    }

                    # Save to file (one file per trajectory, timestamp prefix for easy sorting)
                    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tid}.json")
                    with open(log_file, "w") as f:
                        json.dump(trajectory_log, f, indent=2, ensure_ascii=False)
                    print(f"[{tid}] Trajectory saved to {log_file}")

                except Exception as e:
                    print(f"[{tid}] Failed to save trajectory log: {e}")

                # CRITICAL: rollout_infos must be JSON-serializable
                # Do NOT include trajectory_info - it contains large base64 image data
                # that causes serialization issues when sent to trainer
                rollout_infos = {
                    "num_turns": current_step + 1,
                    "trajectory_id": trajectory_id,
                    "total_reward": float(total_reward),
                    "step_rewards": [float(r) for r in step_rewards],
                    "reward_source": "mcp_step_sum",
                    "done": bool(done),
                    "task_id": info.get("task_id", task_id or ""),  # for GRPO grouping verification
                    "success": bool(
                        done
                        and any(s.get("reward_breakdown", {}).get("completion_rate") == 1.0 for s in trajectory_steps)
                    ),
                }

                # Verify serialization (silent unless error)
                try:
                    json.dumps(rollout_infos)
                except Exception as e:
                    print(f"[{tid}] ERROR: rollout_infos serialization FAILED: {e}")

                # SUCCESS! Return the completed rollout
                trajectory_completed = True
                return RolloutOutput(
                    response=response,
                    messages=final_messages,
                    rollout_infos=rollout_infos,
                )

            finally:
                # Release semaphore FIRST so other waiting rollouts can start connecting.
                if _semaphore_acquired:
                    self._rollout_semaphore.release()
                    print(f"[{tid}] Semaphore released")

                # Port handling.
                # - Normal completion: synchronous put_nowait so the next rollout
                #   can grab it immediately.
                # - Abnormal exit (timeout / cancelled / exception): the
                #   server-side episode lock is almost certainly stuck. Do NOT
                #   recycle the port directly; instead schedule an in-place
                #   restart that will recycle the port once the new MCP server
                #   process is back to HTTP 200. This preserves (DSW, task_id)
                #   pairing required by GRPO grouping.
                if server_url and dsw_idx is not None and dsw_idx in self._round_dsw_port_queues:
                    if trajectory_completed:
                        self._round_dsw_port_queues[dsw_idx].put_nowait(server_url)
                        print(f"[{tid}] Port released back to DSW[{dsw_idx}] queue")
                    else:
                        print(
                            f"[{tid}] Trajectory aborted; scheduling MCP restart on {server_url} "
                            f"(port held out of queue until ready)"
                        )
                        try:
                            loop = asyncio.get_event_loop()
                            loop.create_task(
                                self.server_pool.restart_and_recycle_port(
                                    dsw_idx,
                                    server_url,
                                    port_queue=self._round_dsw_port_queues[dsw_idx],
                                    ready_timeout=300.0,
                                )
                            )
                        except Exception as e:
                            print(f"[{tid}] Failed to schedule restart: {e}; recycling port as best-effort")
                            try:
                                self._round_dsw_port_queues[dsw_idx].put_nowait(server_url)
                            except Exception:
                                pass
                elif server_url:
                    if dsw_idx is not None:
                        await self.server_pool.release_port(dsw_idx, server_url)
                    else:
                        await self.server_pool.release(server_url)
                    print(f"[{tid}] Server released (fallback)")

                # Disconnect SSE AFTER port release, with asyncio.shield() so that
                # CancelledError from the outer task does NOT abort the disconnect —
                # without shield, CancelledError propagates through wait_for and
                # leaves MCP session internals in a torn state.
                if env is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(env.disconnect()), timeout=8.0)
                    except BaseException:
                        pass

        def _create_failed_trajectory_output(self, infer_request, error_msg: str):
            """Create a minimal RolloutOutput for a failed trajectory that actually started execution.

            This should ONLY be used when:
            - Task was successfully received
            - At least one action was attempted
            - Failure occurred mid-execution

            NOT for infrastructure failures like connection timeouts.
            """
            from swift.llm.infer.protocol import RolloutOutput

            # Create minimal failed trajectory with penalty reward
            failed_messages = [
                {"role": "user", "content": "Task failed to execute"},
                {"role": "assistant", "content": f"Error: {error_msg}"},
            ]

            rollout_infos = {
                "num_turns": 1,
                "trajectory_id": infer_request.uuid,
                "total_reward": -1.0,  # Penalty for failed trajectory
                "step_rewards": [-1.0],
                "done": True,
                "error": error_msg,
            }

            print(f"[MCPMultiTurnScheduler] Created failed trajectory output with reward -1.0")

            return RolloutOutput(
                response=None,
                messages=failed_messages,
                rollout_infos=rollout_infos,
            )

        def _create_infrastructure_failure_output(self, infer_request, num_retries: int):
            """Create a fallback RolloutOutput when all trajectory attempts fail due to infrastructure issues.

            This prevents the trainer from crashing while still providing a signal that something went wrong.
            The reward is set to 0 (neutral) to avoid polluting training with fake negative signals
            that aren't the model's fault.
            """
            from swift.llm.infer.protocol import (
                RolloutOutput,
                ChatCompletionResponse,
                ChatCompletionResponseChoice,
                ChatMessage,
                UsageInfo,
            )

            error_msg = f"Infrastructure failure after {num_retries} attempts"
            fallback_content = '<think>Infrastructure error occurred.</think>\n{"tool_name": "done", "args": {}}'

            fallback_messages = [
                {"role": "user", "content": "System: Task could not be executed due to infrastructure issues"},
                {"role": "assistant", "content": fallback_content},
            ]

            rollout_infos = {
                "num_turns": 1,
                "trajectory_id": infer_request.uuid,
                "total_reward": 0.0,
                "step_rewards": [0.0],
                "done": True,
                "error": error_msg,
                "infrastructure_failure": True,
            }

            dummy_response = ChatCompletionResponse(
                model="",
                choices=[
                    ChatCompletionResponseChoice(
                        index=0, message=ChatMessage(role="assistant", content=fallback_content), finish_reason="stop"
                    )
                ],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

            print(f"[MCPMultiTurnScheduler] Created infrastructure failure output with reward 0.0 (neutral)")

            return RolloutOutput(
                response=dummy_response,
                messages=fallback_messages,
                rollout_infos=rollout_infos,
            )

    # Register the scheduler
    multi_turns["mcp_scheduler"] = MCPMultiTurnScheduler
    print("[MCPMultiTurnScheduler] Registered as 'mcp_scheduler' in multi_turns")

except ImportError as e:
    print(f"[MCPMultiTurnScheduler] Warning: Could not register scheduler: {e}")
