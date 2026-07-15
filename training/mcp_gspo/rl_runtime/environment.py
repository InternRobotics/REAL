"""REAL MCP environment, reward state, and ms-swift environment adapter."""

import gymnasium as gym
import numpy as np
import json
import re
import sys
import time
from copy import deepcopy
import random
import asyncio
from typing import Optional, Tuple, Dict, Any, List, Union
from abc import ABC
import os

from rl_runtime.real_contract import (
    canonical_tool_name,
    normalize_server_arguments,
    resolve_server_tool,
    validate_real_action_tools,
)
from rl_runtime.action_parser import ActionParseError, parse_action
from rl_runtime.mcp_diff_metric import MCPDiffMetric
from rl_runtime.mcp_diff_reward import MCPAntiExploitReward, MCPDiffReward, MCPDiffRewardWithPartialCredit
from rl_runtime.prompt import load_prompt_template

# Ensure localhost/127.0.0.1 bypasses HTTP proxy (SSH tunnels must not go through proxy)
_existing_no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
_no_proxy_entries = set(e.strip() for e in _existing_no_proxy.split(",") if e.strip())
_no_proxy_entries.update(["localhost", "127.0.0.1"])
os.environ["no_proxy"] = ",".join(_no_proxy_entries)
os.environ["NO_PROXY"] = os.environ["no_proxy"]


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runtime_path(env_name: str, *relative_parts: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    return os.path.join(_BASE_DIR, *relative_parts)


def should_treat_reset_error_as_task_not_found(observation: str, info: dict) -> bool:
    """Detect reset failures caused by hitting a port whose scene does not contain the task."""
    observation_text = str(observation or "").lower()
    instruction_text = str((info or {}).get("instruction", "")).lower()
    return "not found" in observation_text or "not found" in instruction_text


# MCP Client imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, ImageContent

# Register Env for swift
from swift.plugin.env import Env as SwiftEnvBase, envs


# =============================================================================

# =============================================================================


class HistoryManager:
    def __init__(self):
        self.accumulated_history: List[str] = []
        self.step_num = 0

    def reset(self):
        self.accumulated_history = []
        self.step_num = 0

    def extract_new_step_from_gpt_output(self, gpt_history: str) -> Optional[str]:
        if not gpt_history:
            return None

        lines = gpt_history.strip().split("\n")
        if not lines:
            return None

        last_line = lines[-1].strip()

        match = re.match(r"^\d+\)\s*(.+)$", last_line)
        if match:
            return match.group(1).strip()

        return last_line

    def add_step(self, description: str):
        if not description:
            return
        formatted_step = f"{self.step_num}) {description.strip()}"
        self.accumulated_history.append(formatted_step)
        self.step_num += 1

    def get_formatted_history(self) -> str:
        return "\n".join(self.accumulated_history)

    def get_task_progress_str(self) -> str:
        if not self.accumulated_history:
            return "{}"
        else:
            history_str = self.get_formatted_history()
            return f"{{'History': '{history_str}'}}"

    def __repr__(self):
        return f"HistoryManager(step_num={self.step_num}, history_length={len(self.accumulated_history)})"


class MCPSingleEnv:
    def __init__(self, env_id: int, server_url: str, env_config: Dict = None):
        self.env_id = env_id
        self.server_url = server_url
        self.env_config = env_config or {}

        # State
        self.current_step = 0
        self.instruction = ""
        self.world_graph = {}
        self.target_world_graph = {}
        self.current_inv = None
        self.current_marker_map = None  # Track marker assignments for reward calculation
        self.is_done = False
        self.hand_occupied = False  # Track whether robot hand holds an object

        # Disambiguation chain tracking (distractor → ask → use marker)
        self._distractor_visible = False  # multiple target-category objects in current marker map
        self._last_ask_marker = None  # marker_id extracted from most recent ask response
        self._last_ask_target = None  # target description extracted from most recent ask response
        self._asked_since_ambiguity = False  # ask was called after _distractor_visible was set

        # UID to category mapping for reward calculation
        self.uid_to_category = {}  # Maps UID (e.g., '_37cecdef...') to category (e.g., 'remote control')

        # MCP tools (dynamically fetched from server)
        self.mcp_tools = []  # List of Tool objects from server

        # Reward and metrics
        # Default to partial credit (includes ask reward) unless explicitly disabled
        reward_config = self.env_config.get("reward_config", {})
        use_partial_credit = reward_config.get("use_partial_credit", True)
        use_anti_exploit = reward_config.get("use_anti_exploit", False)

        if use_anti_exploit:
            self.reward_func = MCPAntiExploitReward(reward_config)
        elif use_partial_credit:
            self.reward_func = MCPDiffRewardWithPartialCredit(reward_config)
        else:
            self.reward_func = MCPDiffReward(reward_config)

        metric_config = self.env_config.get("metric_config", {})
        self.metric_tracker = MCPDiffMetric(metric_config)
        self.episode_count = 0

        # Config
        self.max_steps = self.env_config.get("max_steps", 50)

        # Persistent SSE connection (set by connect(), cleared by disconnect())
        self._sse_cm = None  # sse_client context manager
        self._sse_streams = None  # (read_stream, write_stream) tuple
        self._session_cm = None  # ClientSession context manager
        self._session = None  # Active ClientSession instance
        self._connected = False

    async def connect(self):
        """Establish persistent SSE + MCP session to the server.

        Call once before reset()/step() to avoid per-call reconnection overhead.
        Safe to call when already connected (no-op).
        """
        if self._connected:
            return
        await self._do_connect()

    async def _do_connect(self):
        """Internal: open SSE stream and MCP session."""
        # Clean up any partial state first
        await self._do_disconnect()

        try:
            self._sse_cm = sse_client(self.server_url, timeout=60)
            self._sse_streams = await self._sse_cm.__aenter__()
            read_stream, write_stream = self._sse_streams

            self._session_cm = ClientSession(read_stream, write_stream)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
            self._connected = True
            print(f"[MCPSingleEnv#{self.env_id}] Persistent session established -> {self.server_url}")
        except Exception as e:
            print(f"[MCPSingleEnv#{self.env_id}] Failed to establish persistent session: {e}")
            await self._do_disconnect()
            raise

    async def disconnect(self):
        """Close the persistent SSE + MCP session.

        Safe to call when not connected (no-op).
        """
        if not self._connected and self._session is None and self._sse_cm is None:
            return
        await self._do_disconnect()

    async def _do_disconnect(self):
        """Internal: tear down session and SSE stream, swallowing errors."""
        self._connected = False
        self._session = None

        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None

        if self._sse_cm is not None:
            try:
                await self._sse_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._sse_cm = None

        self._sse_streams = None

    async def _ensure_connected(self):
        """Reconnect if the persistent session is down."""
        if not self._connected or self._session is None:
            await self._do_connect()

    def build_tools_description(self) -> str:
        """
        Build tools description from MCP server's registered tools.

        Returns:
            Formatted string describing all available tools
        """
        if not self.mcp_tools:
            return "No tools available."

        lines = []
        for tool in self.mcp_tools:
            # Extract parameter info from inputSchema
            params = []
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                schema = tool.inputSchema
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    if param_name in required:
                        params.append(f"{param_name}")
                    else:
                        params.append(f"[{param_name}]")

            # Format: tool_name(params): description
            param_str = ", ".join(params) if params else ""
            desc = tool.description[:80] if len(tool.description) > 80 else tool.description
            lines.append(f"- {tool.name}({param_str}): {desc}")

        return "\n".join(lines)

    def _build_uid_to_category_mapping(self, reset_info: Dict) -> Dict[str, str]:
        """
        Build UID to category mapping from reset_info.

        Uses:
        - target_object: {"id": "xxx", "category": "remote control"}
        - distractor_metadata: {"_xxx": {"category": "box"}, ...}
        - object_distractors: ["_xxx", "_yyy"]

        Returns:
            Dict mapping UID prefixes to category names
        """
        uid_to_cat = {}

        # Extract from target_object
        target_obj = reset_info.get("target_object", {})
        if target_obj:
            obj_id = target_obj.get("id", "")
            category = target_obj.get("category", "")
            if obj_id and category:
                # Store both with and without leading underscore
                uid_to_cat[obj_id] = category
                uid_to_cat[f"_{obj_id}"] = category
                # Also handle full name format: {id}_on_{furniture}_at_{task_id}
                # We'll match by prefix

        # Extract from distractor_metadata
        distractor_meta = reset_info.get("distractor_metadata", {})
        for uid, meta in distractor_meta.items():
            if isinstance(meta, dict):
                category = meta.get("category", "")
                if category:
                    uid_to_cat[uid] = category
                    # Handle without leading underscore too
                    if uid.startswith("_"):
                        uid_to_cat[uid[1:]] = category

        return uid_to_cat

    def _convert_world_graph_to_category(self, world_graph: Dict) -> Dict:
        """
        Convert world_graph from UIDs to category names.

        Input format:
            {'bed_1': {'content': ['_xxx_on_bed_1_at_task', '_yyy_on_bed_1_at_task']}}

        Output format:
            {'bed_1': {'content': ['remote control', 'box']}}
        """
        converted = {}
        for furniture, payload in world_graph.items():
            if not isinstance(payload, dict):
                converted[furniture] = payload
                continue

            content = payload.get("content", [])
            converted_content = []
            for obj_name in content:
                # Extract UID from full name: {uid}_on_{furniture}_at_{task_id}
                if "_on_" in obj_name and "_at_" in obj_name:
                    uid_part = obj_name.split("_on_")[0]
                else:
                    uid_part = obj_name

                # Look up category
                category = None
                # Try exact match first
                if uid_part in self.uid_to_category:
                    category = self.uid_to_category[uid_part]
                else:
                    # Try matching by prefix (for UIDs that might have extra characters)
                    for uid, cat in self.uid_to_category.items():
                        if uid_part.startswith(uid) or uid.startswith(uid_part):
                            category = cat
                            break

                if category:
                    converted_content.append(category)
                else:
                    # Fallback to original if no mapping found
                    converted_content.append(obj_name)

            converted[furniture] = {"content": converted_content}
            # Preserve other keys like 'door'
            for key, value in payload.items():
                if key != "content":
                    converted[furniture][key] = value

        return converted

    def _validate_world_graph(self, new_wg: Dict, tool_name: str) -> Tuple[bool, str]:
        """
        Validate a new world_graph against the current one to detect server anomalies.

        Checks:
        1. Unknown objects: UIDs that couldn't be mapped to categories (contain hex chars or '_on_' suffix)
        2. Excessive drift: too many objects appeared/disappeared for a single action
        3. Object teleportation: objects appearing on furniture they were never on

        Returns:
            (is_valid, reason) — if invalid, reason describes the anomaly
        """
        if not self.world_graph or not new_wg:
            return True, ""

        # Count unknown objects (raw UIDs that failed category mapping)
        unknown_objects = []
        for furniture, payload in new_wg.items():
            for obj in payload.get("content", []):
                # Heuristic: raw UIDs contain hex chars or long hash-like patterns
                # Category names are human-readable (e.g. "decorative object", "remote control")
                if "_on_" in obj and "_at_" in obj:
                    # Still has runtime suffix — _convert_world_graph_to_category failed
                    unknown_objects.append((obj, furniture))
                elif len(obj) > 30 and all(c in "0123456789abcdef_-" for c in obj.replace("-", "")):
                    # Looks like a raw UUID/hash
                    unknown_objects.append((obj, furniture))

        if unknown_objects:
            return False, (
                f"Unknown objects in world_graph (UID mapping failed): "
                f"{unknown_objects[:3]}{'...' if len(unknown_objects) > 3 else ''}"
            )

        # Count total object changes
        old_total = sum(len(p.get("content", [])) for p in self.world_graph.values() if isinstance(p, dict))
        new_total = sum(len(p.get("content", [])) for p in new_wg.values() if isinstance(p, dict))
        delta = abs(new_total - old_total)

        # A single action should not cause more than ~3 objects to change
        # (pick removes 1, place adds 1, open/close might shift a few)
        MAX_SINGLE_STEP_DELTA = 3
        if delta > MAX_SINGLE_STEP_DELTA and tool_name not in ("finish", "finish_with_id", ""):
            return False, (
                f"Excessive world_graph drift: {old_total} -> {new_total} objects (delta={delta}) after '{tool_name}'"
            )

        return True, ""

    def _check_observation_for_errors(self, observation: str, tool_name: str) -> Tuple[float, Dict]:
        """Check observation text for MCP server error indicators and return penalty."""
        if not observation:
            return 0.0, {}

        obs_lower = observation.lower()

        severe_patterns = [
            "[[internal error]]",
            "error executing",
            "traceback",
            "no object in the inventory",
            "cannot place object into a closed",
        ]
        moderate_patterns = [
            "not found in the marker map",
            "cannot find a non-occluded position",
            "not found in the scene",
            "not implemented yet",
            "not recognized",
            "not aligned to bread target",
            "selected marker_id",
            "hand is empty",
        ]

        for pattern in severe_patterns:
            if pattern in obs_lower:
                print(f"[MCPSingleEnv] Tool error (severe): '{pattern}' detected, penalty: -0.5")
                return -0.5, {"error_type": "severe", "error_pattern": pattern}

        for pattern in moderate_patterns:
            if pattern in obs_lower:
                print(f"[MCPSingleEnv] Tool error (moderate): '{pattern}' detected, penalty: -0.3")
                return -0.3, {"error_type": "moderate", "error_pattern": pattern}

        return 0.0, {}

    def _check_hand_constraint(self, tool_name: str) -> Tuple[float, Dict]:
        """Check if tool call violates the hand occupancy constraint.

        Rules:
        - pick, open, close: require hand EMPTY
        - place: requires hand OCCUPIED
        """
        if tool_name in ("pick", "open", "close") and self.hand_occupied:
            print(
                f"[MCPSingleEnv] Hand violation: {tool_name} while hand occupied (inv={self.current_inv}), penalty: -0.3"
            )
            return -0.3, {
                "violation_type": f"{tool_name}_while_hand_occupied",
                "hand_state": "occupied",
                "current_inv": str(self.current_inv),
            }
        elif tool_name == "place" and not self.hand_occupied:
            print(f"[MCPSingleEnv] Hand violation: place while hand empty, penalty: -0.3")
            return -0.3, {
                "violation_type": "place_while_hand_empty",
                "hand_state": "empty",
            }
        return 0.0, {}

    async def reset(self, seed: int = None, task_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment by calling finish/finish_with_id to get new task.

        Uses the persistent SSE session (connect() should be called first).
        Falls back to one-shot connection if no persistent session exists.

        Args:
            seed: Random seed
            task_id: If provided, use finish_with_id(task_id) to load specific task.
                     If None, use finish() to get next sequential task.
        """
        if seed is not None:
            random.seed(seed + self.env_id)  # Different seed per env

        self.current_step = 0
        self.is_done = False
        self.hand_occupied = False
        self.current_inv = None
        self._distractor_visible = False
        self._last_ask_marker = None
        self._last_ask_target = None
        self._asked_since_ambiguity = False

        # Retry logic with exponential backoff for connection failures
        max_retries = 5
        base_delay = 1.0  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                # Use persistent session if available, otherwise one-shot
                await self._ensure_connected()
                session = self._session

                # Fetch available tools from server (dynamic tool discovery)
                tools_response = await session.list_tools()
                self.mcp_tools = tools_response.tools
                print(f"[MCPSingleEnv] Discovered {len(self.mcp_tools)} tools from server")

                contract_errors = validate_real_action_tools(self.mcp_tools)
                if contract_errors:
                    message = "; ".join(contract_errors)
                    if os.environ.get("MCP_STRICT_REAL_CONTRACT", "1") == "1":
                        raise RuntimeError(f"MCP server is not REAL-compatible: {message}")
                    print(f"[MCPSingleEnv] REAL contract warning: {message}")

                # Call finish or finish_with_id to get task info
                if task_id is not None:
                    print(f"[MCPSingleEnv] Using finish_with_id(task_id={task_id})")
                    response = await session.call_tool("finish_with_id", {"task_id": task_id})
                else:
                    response = await session.call_tool("finish", {})

                if len(response.content) == 0:
                    self.is_done = True
                    return "No response from server.", {"error": True, "env_id": self.env_id}

                # Safe text extraction: server may return ImageContent or other
                # non-text content if the tool errored mid-flight (e.g. Isaac
                # Sim returned a screenshot during a transient failure). Walk
                # contents and pick the first TextContent; fall back to repr if
                # none has .text.
                first = response.content[0]
                response_text = getattr(first, "text", None)
                if response_text is None:
                    text_item = next(
                        (c for c in response.content if isinstance(c, TextContent)),
                        None,
                    )
                    if text_item is not None:
                        response_text = text_item.text
                    else:
                        # Hard fallback: surface the type so logs are debuggable
                        # rather than crash with `'ImageContent' has no .text`.
                        types = [type(c).__name__ for c in response.content]
                        print(f"[MCPSingleEnv] Reset response has no TextContent (got {types}); treating as error.")
                        self.is_done = True
                        return f"No text in reset response (types={types}).", {"error": True, "env_id": self.env_id}

                if (
                    "All benchmark episodes completed" in response_text
                    or "All episodes completed" in response_text
                    or "All replay episodes completed" in response_text
                ):
                    self.is_done = True
                    return "All episodes completed.", {"done": True, "env_id": self.env_id}

                # Parse JSON response
                scene_description = ""
                all_furnitures = []
                reset_info = {}
                try:
                    reset_info = json.loads(response_text)
                    self.instruction = reset_info.get("task_description", "")
                    self.world_graph = reset_info.get("initial_world_graph", {})
                    self.target_world_graph = reset_info.get("goal_world_graph") or reset_info.get(
                        "target_world_graph", {}
                    )
                    episode_id = (
                        reset_info.get("episode_id")
                        or reset_info.get("episode_idx")
                        or reset_info.get("task_id", self.episode_count)
                    )

                    rooms_and_furniture = reset_info.get("rooms_and_furniture", {})
                    if rooms_and_furniture:
                        scene_description = "## Scene Description\nThe scene has the following rooms and receptacles:\n"
                        for idx, (room, furnitures) in enumerate(rooms_and_furniture.items(), 1):
                            furniture_list = furnitures if isinstance(furnitures, list) else [furnitures]
                            scene_description += f"{idx}) In {room}: {', '.join(furniture_list)}.\n"
                            all_furnitures.extend(furniture_list)
                    else:
                        scene_description = reset_info.get("scene_description", "")
                        all_furnitures = reset_info.get("all_furnitures", [])

                    self.target_object_info = reset_info.get("target_object", {})
                    self.source_furniture = reset_info.get("source_furniture", "")
                    self.destination_furniture = reset_info.get("destination_furniture", "")

                    self.uid_to_category = self._build_uid_to_category_mapping(reset_info)
                    print(f"[MCPSingleEnv] UID to category mapping: {self.uid_to_category}")

                    self.world_graph = self._convert_world_graph_to_category(self.world_graph)
                    self.target_world_graph = self._convert_world_graph_to_category(self.target_world_graph)

                    print(f"[MCPSingleEnv] Source furniture: {self.source_furniture}")
                    print(f"[MCPSingleEnv] Destination furniture: {self.destination_furniture}")
                    print(f"[MCPSingleEnv] Target object: {self.target_object_info}")
                    print(f"[MCPSingleEnv] Initial world_graph: {json.dumps(self.world_graph)}")
                    print(f"[MCPSingleEnv] Target world_graph: {json.dumps(self.target_world_graph)}")

                    # Validate that the server actually loaded the requested task
                    if task_id is not None:
                        resp_task_id = (
                            reset_info.get("task_id") or reset_info.get("episode_id") or reset_info.get("episode_idx")
                        )
                        print(f"[MCPSingleEnv] Response task_id={resp_task_id} (requested={task_id})")
                        if resp_task_id and str(resp_task_id) != str(task_id):
                            print(
                                f"[MCPSingleEnv] ⚠️ Task ID mismatch: requested={task_id}, got={resp_task_id} (stale SSE data?)"
                            )
                            self.is_done = True
                            return f"Invalid task (task_id mismatch: got {resp_task_id}).", {
                                "error": True,
                                "env_id": self.env_id,
                                "invalid_task": True,
                                "stale_sse": True,
                                "instruction": f"Task ID mismatch: requested {task_id}, got {resp_task_id}",
                            }

                except json.JSONDecodeError:
                    self.instruction = response_text
                    if "Task:" in response_text:
                        task_start = response_text.find("Task:") + 5
                        task_end = (
                            response_text.find("Instruction:")
                            if "Instruction:" in response_text
                            else len(response_text)
                        )
                        self.instruction = response_text[task_start:task_end].strip()

                    if len(response.content) > 1:
                        try:
                            self.world_graph = json.loads(response.content[1].text)
                        except json.JSONDecodeError:
                            self.world_graph = {}
                    else:
                        self.world_graph = {}

                    self.target_world_graph = self.env_config.get("target_world_graph", {})
                    episode_id = self.episode_count

                self.reward_func.reset(initial_wg=self.world_graph, goal_wg=self.target_world_graph)

                target_diff_count = len(self.reward_func.get_target_diffs())

                # Validate task: target_diffs must be > 0 and world_graph non-empty
                # Invalid tasks (e.g., ask reply leaked as task_description) have
                # target_diffs=0 and empty world_graph, producing false successes.
                if target_diff_count == 0 or not self.world_graph:
                    print(
                        f"[MCPSingleEnv] ⚠️ Invalid task: target_diffs={target_diff_count}, "
                        f"world_graph_empty={not self.world_graph}, "
                        f"instruction={self.instruction}"
                    )
                    self.is_done = True
                    return "Invalid task (no target diffs).", {
                        "error": True,
                        "env_id": self.env_id,
                        "invalid_task": True,
                        "instruction": self.instruction,
                    }

                self.metric_tracker.reset_episode(episode_id=episode_id, target_diff_count=target_diff_count)
                self.episode_count += 1

                observation = f"Instruction: {self.instruction}"
                info = {
                    "instruction": self.instruction,
                    "target_diffs": target_diff_count,
                    "world_graph": self.world_graph,
                    "target_world_graph": self.target_world_graph,
                    "env_id": self.env_id,
                    "scene_description": scene_description,
                    "all_furnitures": all_furnitures,
                    "target_object": self.target_object_info if hasattr(self, "target_object_info") else {},
                    "source_furniture": self.source_furniture if hasattr(self, "source_furniture") else "",
                    "destination_furniture": self.destination_furniture
                    if hasattr(self, "destination_furniture")
                    else "",
                    "task_id": reset_info.get("task_id", str(episode_id)),
                }

                return observation, info

            except Exception as e:
                last_error = e
                # Connection is likely broken — tear it down so next retry reconnects
                await self._do_disconnect()
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"[MCPSingleEnv] Connection failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    import traceback

                    traceback.print_exc()

        return f"Connection error after {max_retries} retries: {last_error}", {"error": True, "env_id": self.env_id}

    def _update_distractor_state(self, tool_name: str) -> None:
        """Set _distractor_visible when a perception call reveals multiple target-category objects."""
        if canonical_tool_name(tool_name) not in ("explore_receptacle", "find_objects"):
            return
        if not self.current_marker_map or not self.target_object_info:
            return
        toi = self.target_object_info
        if isinstance(toi, dict):
            target_cat = toi.get("category", "").lower()
        elif isinstance(toi, list) and toi:
            first = toi[0]
            target_cat = (first.get("category", "") if isinstance(first, dict) else str(first)).lower()
        else:
            return
        if not target_cat:
            return
        count = sum(
            1
            for v in self.current_marker_map.values()
            if target_cat in str(v.get("category", v.get("name", v)) if isinstance(v, dict) else v).lower()
        )
        if count > 1:
            if not self._distractor_visible:  # only reset on fresh detection
                self._asked_since_ambiguity = False
            self._distractor_visible = True
            print(f"[MCPSingleEnv] Distractor detected: {count} × '{target_cat}' in marker map")

    def _extract_marker_from_ask_response(self, observation: str) -> Optional[str]:
        """Extract a marker_id from ask response, validated against current_marker_map."""
        if not observation:
            return None
        # Prefer exact match against known markers — longest first to avoid substring false matches
        if self.current_marker_map:
            for marker_id in sorted(self.current_marker_map, key=len, reverse=True):
                if marker_id in observation:
                    return marker_id
        # Fallback: loose regex, capture group(1) = the ID part only, then validate
        m = re.search(r"marker[_\s](\w+)", observation, re.IGNORECASE)
        if m:
            raw = m.group(1)
            if not self.current_marker_map or raw in self.current_marker_map:
                return raw
        return None

    def _extract_target_from_ask_response(self, observation: str) -> Optional[str]:
        """Legacy fallback retained for old callers.

        REAL's social-agent contract returns a marker id, so guessing a target
        from hard-coded category keywords is both brittle and unnecessary.
        """
        return None

    async def step(self, action_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute action through MCP server with global timeout"""
        if self.is_done:
            return "Episode already done.", 0.0, True, {"env_id": self.env_id}

        # Global timeout for the entire step (120 seconds)
        STEP_TIMEOUT = 150.0
        try:
            return await asyncio.wait_for(self._step_impl(action_text), timeout=STEP_TIMEOUT)
        except asyncio.TimeoutError:
            self.current_step += 1
            print(f"[MCPSingleEnv] ⚠️ STEP TIMEOUT after {STEP_TIMEOUT}s! Action: {action_text[:100]}...")

            # Return informative error for VLM to learn replan
            timeout_observation = (
                f"⚠️ Action timeout after {STEP_TIMEOUT} seconds. "
                f"The action '{action_text[:50]}...' took too long to execute. "
                f"This may be due to network issues or server overload. "
                f"Please try a different approach or retry the action."
            )

            # Don't terminate episode - let VLM try to recover
            # Give small penalty but allow replan
            timeout_penalty = -0.3

            # IMPORTANT: Sync reward_func.prev_world_graph even on timeout.
            # Without this, the next successful step will compute diff against
            # a stale prev_world_graph, causing rewards to be attributed to the
            # wrong action (e.g. focus_on getting pick's progress reward).
            if hasattr(self, "reward_func") and self.reward_func is not None:
                # Call compute_reward with current (unchanged) world_graph
                # so that prev_world_graph is updated and any server-side
                # state changes that leaked through are accounted for NOW.
                _, timeout_reward_info = self.reward_func.compute_reward(
                    self.world_graph, robot_inv=None, marker_map=self.current_marker_map
                )
                print(
                    f"[MCPSingleEnv] Timeout: synced reward_func prev_world_graph "
                    f"(completion={timeout_reward_info.get('completion_rate', 'N/A')})"
                )

            self.metric_tracker.update_step(reward=timeout_penalty, info={"timeout": True})

            return (
                timeout_observation,
                timeout_penalty,
                False,
                {
                    "env_id": self.env_id,
                    "timeout": True,
                    "recoverable": True,
                    "server_url": self.server_url,
                    "raw_completion": action_text[:500],
                    "suggestion": "Consider trying a different action or retrying",
                },
            )

    async def _step_impl(self, action_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Internal step implementation (wrapped by timeout in step())"""
        self.current_step += 1
        reward = 0.0
        done = False
        info = {"env_id": self.env_id}

        # Log raw VLM output (truncated) for debugging
        info["raw_completion"] = action_text[:500]

        try:
            parsed_action = parse_action(action_text)
            requested_tool_name = parsed_action.requested_tool_name
            tool_name = parsed_action.tool_name
            tool_args = parsed_action.arguments
            info["requested_tool_name"] = requested_tool_name
        except ActionParseError as e:
            # Invalid action format - give penalty but don't terminate episode
            # This allows the model to recover and try again
            penalty = -0.5
            self.metric_tracker.update_step(reward=penalty, info={"invalid_action": True})
            print(f"[MCPSingleEnv] Invalid action format: {e}")
            return (
                f'Invalid action format. Please output JSON: {{"tool_name": "...", "args": {{...}}}}. Error: {e}',
                penalty,
                False,
                {"error": str(e), "invalid_action": True, "env_id": self.env_id},
            )

        # Handle finish specially
        if tool_name == "finish":
            done = True
            self.is_done = True
            if self.reward_func.is_done():
                reward = self.reward_func.reward_weights.get("perfect_match", 2.0)
                observation = "Task completed successfully!"
            else:
                completion = self.reward_func.get_completion_rate()
                # Apply premature_finish penalty directly here — track_action is never reached
                # because this branch returns early, so the -3.0 defined there was never executed.
                premature_penalty = -3.0
                if hasattr(self.reward_func, "anti_exploit_weights"):
                    premature_penalty = self.reward_func.anti_exploit_weights.get("premature_finish", -3.0)
                reward = -1.0 + completion + premature_penalty
                observation = f"Task incomplete. Completion: {completion:.1%}"
                print(f"[MCPSingleEnv] Premature finish penalty: {premature_penalty:.1f} (completion={completion:.1%})")

            completion_rate = self.reward_func.get_completion_rate()
            terminal_reward_info = {
                "completion_rate": completion_rate,
                "terminal_reward": reward,
            }
            info["reward_breakdown"] = terminal_reward_info
            info["completion_rate"] = completion_rate
            self.metric_tracker.update_step(reward=reward, info=terminal_reward_info, terminated=True)
            self.metric_tracker.end_episode(
                success=self.reward_func.is_done(),
                timeout=False,
                completion_rate=completion_rate,
            )
            return observation, reward, done, info

        # Check hand constraint BEFORE execution
        hand_penalty, hand_info = self._check_hand_constraint(tool_name)
        if hand_penalty != 0:
            reward += hand_penalty
            info["hand_violation"] = hand_info

        # Execute tool through MCP with retry logic
        max_retries = 5
        base_delay = 1.0
        last_error = None
        info["server_url"] = self.server_url

        for attempt in range(max_retries):
            try:
                # Use persistent session if available, otherwise reconnect
                await self._ensure_connected()
                session = self._session

                try:
                    server_tool_name = resolve_server_tool(tool_name, self.mcp_tools)
                    server_tool_args = normalize_server_arguments(tool_name, tool_args, self.mcp_tools)
                    result = await session.call_tool(server_tool_name, server_tool_args)
                    tool_args = server_tool_args
                except Exception as e:
                    self.is_done = True
                    return f"Tool error: {e}", -1.0, True, {"error": str(e), "env_id": self.env_id}

                # Parse observation and world graph
                # Response format from mcp4sft.py:
                # - Content[0]: Image (for nav/gaze actions) or Text (for walkaround/show)
                # - Content[-1]: WORLD_GRAPH JSON (always appended by return_result)
                # NOTE: For 'ask' tool, response is text in content[0], world_graph in content[-1]
                observation = ""
                images = []  # Store base64 image data

                # For ask tool, we need to capture the answer text
                # The answer might be in any content item before the world_graph
                content_to_parse = result.content[:-1] if len(result.content) > 1 else result.content

                for content in content_to_parse:
                    if isinstance(content, TextContent):
                        # Skip world_graph JSON responses
                        text = content.text.strip()
                        if not text.startswith("{") and not text.startswith("Debug Info:"):
                            observation += text + "\n"
                    elif isinstance(content, ImageContent):
                        # Store image data for VLM
                        images.append({"data": content.data, "mimeType": content.mimeType})
                        observation += "[Image received]\n"

                # Add images to info for caller to use
                if images:
                    info["images"] = images

                # Record tool name and args for tracking (e.g., ask count)
                info["tool_name"] = tool_name
                info["tool_args"] = server_tool_args
                info["server_tool_name"] = server_tool_name

                if getattr(result, "isError", False):
                    error_text = (
                        "\n".join(content.text for content in result.content if isinstance(content, TextContent))
                        or f"MCP tool {server_tool_name} failed"
                    )
                    raise RuntimeError(error_text)

                # For ask tool, also check the last content for answer (in case it's there)
                if tool_name == "ask" and not observation.strip():
                    for content in result.content:
                        if isinstance(content, TextContent):
                            text = content.text.strip()
                            # Skip JSON/debug responses
                            if not text.startswith("{") and not text.startswith("Debug Info:"):
                                observation = text
                                break

                # Provide feedback if no text observation
                if not observation.strip():
                    observation = f"[Action {tool_name} completed]"

                # Store full observation for ask debugging
                if tool_name == "ask":
                    info["ask_response"] = observation

                # Extract world graph update from last content item
                # Server returns debug_info with:
                #   - "world_graph" (lowercase, dict): FULL scene world graph (preferred)
                #   - "WORLD_GRAPH" (uppercase, string): partial, only lists furniture with objects (BUGGY for remove detection)
                # Old local server returns: {"furniture": {"content": [...]}, ...}
                if result.content:
                    # Log MCP response structure for debugging
                    info["mcp_response_types"] = [type(c).__name__ for c in result.content]
                    info["mcp_num_content"] = len(result.content)

                    try:
                        raw_response = result.content[-1].text
                        # Log raw server debug response (truncated)
                        info["mcp_raw_debug"] = raw_response[:1000]
                        # Try to parse as JSON
                        if raw_response.startswith("Debug Info:"):
                            # New server format: "Debug Info:\n{...}"
                            json_part = raw_response.replace("Debug Info:", "").strip()
                            debug_info = json.loads(json_part)

                            # Save server-side state snapshot for debugging
                            info["server_debug"] = {
                                "landmark": debug_info.get("CURRENT_LANDMARK"),
                                "inv": debug_info.get("CURRENT_INV"),
                                "pos": debug_info.get("CURRENT_POS"),
                                "marker_map": debug_info.get("CURRENT_MARKER_MAP"),
                                "world_graph": debug_info.get("world_graph"),
                            }

                            # Extract inventory
                            if "CURRENT_INV" in debug_info:
                                self.current_inv = debug_info["CURRENT_INV"]
                                if self.current_inv:
                                    print(f"[MCPSingleEnv] Robot inventory: {self.current_inv}")

                            # Extract marker map for reward calculation
                            if "CURRENT_MARKER_MAP" in debug_info:
                                self.current_marker_map = debug_info["CURRENT_MARKER_MAP"]

                            # Use full world_graph dict (lowercase) if available — it contains ALL furniture
                            # This avoids the bug where WORLD_GRAPH string omits empty furniture after pick
                            wg_dict = debug_info.get("world_graph", None)
                            if "WORLD_GRAPH" in debug_info:
                                info["grounded_world_graph"] = debug_info.get("WORLD_GRAPH")

                            if wg_dict and isinstance(wg_dict, dict):
                                # Validate it looks like a world graph
                                if any(isinstance(v, dict) and "content" in v for v in wg_dict.values()):
                                    candidate_wg = self._convert_world_graph_to_category(wg_dict)
                                    # Validate against anomalies before accepting
                                    wg_valid, wg_reason = self._validate_world_graph(candidate_wg, tool_name)
                                    if wg_valid:
                                        self.world_graph = candidate_wg
                                        print(
                                            f"[MCPSingleEnv] Updated world_graph (full): {json.dumps(self.world_graph)[:300]}..."
                                        )
                                    else:
                                        print(f"[MCPSingleEnv] ⚠️ REJECTED world_graph update: {wg_reason}")
                                        info["world_graph_rejected"] = wg_reason
                                else:
                                    print(f"[MCPSingleEnv] world_graph dict has unexpected format, skipping")
                            else:
                                # Fallback: parse WORLD_GRAPH string (partial, only furniture with objects)
                                wg_str = debug_info.get("WORLD_GRAPH", "")
                                if wg_str:
                                    updated_wg = {}
                                    for line in wg_str.strip().split("\n"):
                                        if ":" in line:
                                            furniture, content_str = line.split(":", 1)
                                            furniture = furniture.strip()
                                            try:
                                                import ast

                                                content_list = ast.literal_eval(content_str.strip())
                                                if isinstance(content_list, list):
                                                    updated_wg[furniture] = {"content": content_list}
                                            except:
                                                updated_wg[furniture] = {"content": []}

                                    if updated_wg:
                                        converted_update = self._convert_world_graph_to_category(updated_wg)
                                        # Build candidate by merging into current
                                        candidate_wg = {**self.world_graph}
                                        for furniture, payload in converted_update.items():
                                            candidate_wg[furniture] = payload
                                        wg_valid, wg_reason = self._validate_world_graph(candidate_wg, tool_name)
                                        if wg_valid:
                                            self.world_graph = candidate_wg
                                            print(
                                                f"[MCPSingleEnv] Updated world_graph (merged, fallback): {json.dumps(self.world_graph)[:300]}..."
                                            )
                                        else:
                                            print(
                                                f"[MCPSingleEnv] ⚠️ REJECTED world_graph update (fallback): {wg_reason}"
                                            )
                                            info["world_graph_rejected"] = wg_reason
                        elif raw_response.strip():
                            # Old server format: direct JSON world graph
                            updated_wg = json.loads(raw_response)
                            if isinstance(updated_wg, dict):
                                # Extract robot inventory first
                                if "robot_inv" in updated_wg:
                                    self.current_inv = updated_wg.pop("robot_inv")
                                    print(f"[MCPSingleEnv] Robot inventory: {self.current_inv}")
                                # Validate remaining entries are furniture dicts
                                if all(
                                    isinstance(v, dict) and "content" in v
                                    for v in updated_wg.values()
                                    if isinstance(v, dict)
                                ):
                                    candidate_wg = self._convert_world_graph_to_category(updated_wg)
                                    wg_valid, wg_reason = self._validate_world_graph(candidate_wg, tool_name)
                                    if wg_valid:
                                        self.world_graph = candidate_wg
                                        print(
                                            f"[MCPSingleEnv] Converted world_graph: {json.dumps(self.world_graph)[:300]}..."
                                        )
                                    else:
                                        print(f"[MCPSingleEnv] ⚠️ REJECTED world_graph update (old format): {wg_reason}")
                                        info["world_graph_rejected"] = wg_reason
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"[MCPSingleEnv] Failed to parse world_graph: {e}")
                        pass  # Keep existing world_graph if parse fails

                # Check observation for tool errors and apply penalty
                tool_error_penalty, tool_error_info = self._check_observation_for_errors(observation, tool_name)
                if tool_error_penalty != 0:
                    reward += tool_error_penalty
                    info["tool_error"] = tool_error_info

                # Update hand state based on successful actions (no error)
                if tool_error_penalty == 0:
                    if tool_name == "pick":
                        self.hand_occupied = True
                        print(f"[MCPSingleEnv] Hand state: occupied (picked object)")
                        # Ambiguity resolved — clear distractor state so future picks aren't penalised
                        self._distractor_visible = False
                        self._last_ask_marker = None
                        self._last_ask_target = None
                        self._asked_since_ambiguity = False
                    elif tool_name == "place":
                        self.hand_occupied = False
                        print(f"[MCPSingleEnv] Hand state: empty (placed object)")

                # Update disambiguation tracking state
                self._update_distractor_state(tool_name)
                if tool_name == "ask":
                    extracted = self._extract_marker_from_ask_response(observation)
                    if extracted:
                        self._last_ask_marker = extracted
                        self._last_ask_target = None
                        self._asked_since_ambiguity = True
                        print(f"[MCPSingleEnv] Ask marker extracted: {extracted}")
                    else:
                        target = self._extract_target_from_ask_response(observation)
                        if target:
                            self._last_ask_target = target
                            self._asked_since_ambiguity = True
                            print(f"[MCPSingleEnv] Ask target extracted: {target}")

                # Convert robot_inv UID to category name
                current_inv_category = None
                if self.current_inv:
                    # Extract UID from inventory name
                    if "_on_" in str(self.current_inv) and "_at_" in str(self.current_inv):
                        uid_part = str(self.current_inv).split("_on_")[0]
                    else:
                        uid_part = str(self.current_inv)
                    # Look up category
                    current_inv_category = self.uid_to_category.get(uid_part, self.current_inv)

                # Track action for anti-exploit (before computing reward)
                action_tracking_reward = 0.0
                if hasattr(self.reward_func, "track_action"):
                    action_tracking_reward = self.reward_func.track_action(
                        tool_name, tool_args, self.current_marker_map
                    )

                # Save env-level penalties (hand_penalty + tool_error_penalty) accumulated above.
                # compute_reward returns a new value that would overwrite them, so we merge explicitly.
                env_penalties = reward

                # Calculate diff-based reward
                diff_reward, reward_info = self.reward_func.compute_reward(
                    self.world_graph, robot_inv=current_inv_category, marker_map=self.current_marker_map
                )
                reward = diff_reward + env_penalties

                # Add action tracking reward
                reward += action_tracking_reward

                # Calculate tool-specific rewards (find_objects, ask)
                if hasattr(self.reward_func, "compute_tool_reward"):
                    disambiguation_state = {
                        "distractor_visible": self._distractor_visible,
                        "last_ask_marker": self._last_ask_marker,
                        "last_ask_target": self._last_ask_target,
                        "asked_since_ambiguity": self._asked_since_ambiguity,
                        "current_marker_map": self.current_marker_map,
                    }
                    tool_reward, tool_info = self.reward_func.compute_tool_reward(
                        tool_name, tool_args, observation, disambiguation_state
                    )
                    if tool_reward != 0:
                        reward += tool_reward
                        reward_info["tool_reward"] = tool_info
                        print(f"[MCPSingleEnv] Tool reward: {tool_reward:.4f} ({tool_info})")

                # Log reward calculation details
                print(f"[MCPSingleEnv] Step {self.current_step} Reward Calculation:")
                print(f"  - Reward: {reward:.4f}")
                print(f"  - Reward breakdown: {reward_info}")
                print(f"  - Completion rate: {self.reward_func.get_completion_rate():.2%}")
                remaining = self.reward_func.get_remaining_diffs()
                if remaining:
                    print(f"  - Remaining diffs: {remaining[:3]}{'...' if len(remaining) > 3 else ''}")
                sys.stdout.flush()  # Force immediate output

                # Check completion
                if self.reward_func.is_done():
                    done = True
                    self.is_done = True
                    observation += " (All target diffs achieved!)"

                # Check step limit
                if not done and self.current_step >= self.max_steps:
                    done = True
                    self.is_done = True
                    max_steps_penalty = -1.0
                    reward += max_steps_penalty
                    reward_info["max_steps_penalty"] = max_steps_penalty
                    observation += " (Max steps reached)"

                info["reward_breakdown"] = reward_info
                info["completion_rate"] = self.reward_func.get_completion_rate()
                info["hand_occupied"] = self.hand_occupied
                info["world_graph_snapshot"] = self.world_graph
                self.metric_tracker.update_step(reward=reward, info=reward_info, terminated=done)

                if self.reward_func.is_done():
                    self.metric_tracker.end_episode(success=True, timeout=False, completion_rate=1.0)
                elif done:
                    self.metric_tracker.end_episode(
                        success=self.reward_func.is_done(),
                        timeout=True,
                        completion_rate=self.reward_func.get_completion_rate(),
                    )

                # SUCCESS: Return the step result
                return observation.strip(), reward, done, info

            except Exception as e:
                last_error = e
                # Connection is likely broken — tear it down so next retry reconnects
                await self._do_disconnect()
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"[MCPSingleEnv] Step connection failed (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    import traceback

                    traceback.print_exc()

        self.is_done = True
        return (
            f"Connection error after {max_retries} retries: {last_error}",
            -1.0,
            True,
            {"error": str(last_error), "env_id": self.env_id},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "env_id": self.env_id,
            "current_episode": {
                "completion_rate": self.reward_func.get_completion_rate(),
                "remaining_diffs": len(self.reward_func.get_remaining_diffs()),
                "step_count": self.current_step,
            },
            "aggregate": self.metric_tracker.get_metrics(),
        }


class MCPVecEnv:
    def __init__(self, num_envs: int = 1, start_port: int = 8000, host: str = "localhost", env_config: Dict = None):
        self.num_envs = num_envs
        self.start_port = start_port
        self.host = host
        self.env_config = env_config or {}

        # Create environments with different ports
        self.envs: List[MCPSingleEnv] = []
        for i in range(num_envs):
            port = start_port + i
            server_url = f"http://{host}:{port}/sse"
            env = MCPSingleEnv(env_id=i, server_url=server_url, env_config=env_config)
            self.envs.append(env)

        # System prompt (shared)
        self.system_prompt = self._build_system_prompt()

        print(f"[MCPVecEnv] Created {num_envs} environments on ports {start_port}-{start_port + num_envs - 1}")

    def _build_system_prompt(self) -> str:
        return """You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions.

## Tool list
- list_receptacles: List receptacles by room
- navigate_to(receptacle_name): Navigate to a receptacle
- explore_receptacle: Survey objects on the current receptacle
- focus_on(marker_id): Focus on a specific object
- find_objects(target_category): Find objects of a category
- highlight_receptacles: Highlight visible receptacles
- pick(marker_id): Pick up an object
- place(marker_id): Place inventory on a surface
- open(marker_id): Open an articulated door
- close(marker_id): Close an articulated door
- finish: Complete the task
- ask(question): Ask user a question

## Output Format
Output your action as JSON: {"tool_name": "...", "args": {...}}
"""

    def get_server_urls(self) -> List[str]:
        """Get all server URLs for launching MCP servers"""
        return [f"http://{self.host}:{self.start_port + i}/sse" for i in range(self.num_envs)]

    def get_port_range(self) -> Tuple[int, int]:
        """Get port range (start, end) for launching MCP servers"""
        return (self.start_port, self.start_port + self.num_envs)

    async def reset(self, seed: int = None, env_ids: List[int] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Reset environments

        Args:
            seed: Random seed (different per env: seed + env_id)
            env_ids: Specific env IDs to reset. If None, reset all.

        Returns:
            observations: List of observation strings
            infos: List of info dicts
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        # Async gather all resets
        reset_tasks = [self.envs[i].reset(seed=seed) for i in env_ids]
        results = await asyncio.gather(*reset_tasks, return_exceptions=True)

        observations = []
        infos = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                observations.append(f"Error: {result}")
                infos.append({"error": str(result), "env_id": env_ids[i]})
            else:
                obs, info = result
                observations.append(obs)
                infos.append(info)

        return observations, infos

    async def step(
        self, actions: List[str], env_ids: List[int] = None
    ) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        """
        Step all environments with actions

        Args:
            actions: List of action strings (one per env)
            env_ids: Specific env IDs to step. If None, step all.

        Returns:
            observations: List of observation strings
            rewards: List of rewards
            dones: List of done flags
            infos: List of info dicts
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        assert len(actions) == len(env_ids), f"Actions count {len(actions)} != env_ids count {len(env_ids)}"

        # Async gather all steps
        step_tasks = [self.envs[env_ids[i]].step(actions[i]) for i in range(len(actions))]
        results = await asyncio.gather(*step_tasks, return_exceptions=True)

        observations = []
        rewards = []
        dones = []
        infos = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                observations.append(f"Error: {result}")
                rewards.append(-1.0)
                dones.append(True)
                infos.append({"error": str(result), "env_id": env_ids[i]})
            else:
                obs, reward, done, info = result
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

        return observations, rewards, dones, infos

    async def step_with_auto_reset(
        self, actions: List[str], seed: int = None
    ) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        """
        Step all environments and auto-reset any that are done.

        This is the recommended method for RL training loops where you want
        continuous rollouts without manual reset handling.

        Args:
            actions: List of action strings (one per env)
            seed: Random seed for reset (optional)

        Returns:
            observations: List of observations (new task obs for reset envs)
            rewards: List of rewards (from the step that caused done)
            dones: List of done flags (True means env was reset this step)
            infos: List of info dicts, includes 'reset_info' for reset envs
        """
        # Step all envs
        observations, rewards, dones, infos = await self.step(actions)

        # Find envs that need reset
        done_indices = [i for i, done in enumerate(dones) if done]

        if done_indices:
            # Reset done envs
            reset_obs, reset_infos = await self.reset(seed=seed, env_ids=done_indices)

            # Merge reset results into outputs
            for idx, env_idx in enumerate(done_indices):
                # Keep the terminal observation and reward, but add reset info
                infos[env_idx]["terminal_observation"] = observations[env_idx]
                infos[env_idx]["reset_info"] = reset_infos[idx]
                # Replace observation with new task observation
                observations[env_idx] = reset_obs[idx]

        return observations, rewards, dones, infos

    async def reset_done_envs(self, dones: List[bool], seed: int = None) -> Tuple[List[str], List[Dict]]:
        """Reset only the environments that are done"""
        done_env_ids = [i for i, done in enumerate(dones) if done]
        if not done_env_ids:
            return [], []
        return await self.reset(seed=seed, env_ids=done_env_ids)

    def get_active_env_ids(self) -> List[int]:
        """Get IDs of environments that are not done"""
        return [i for i, env in enumerate(self.envs) if not env.is_done]

    def get_done_env_ids(self) -> List[int]:
        """Get IDs of environments that are done"""
        return [i for i, env in enumerate(self.envs) if env.is_done]

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all environments"""
        all_metrics = [env.get_metrics() for env in self.envs]

        # Aggregate
        total_episodes = sum(m["aggregate"]["total_episodes"] for m in all_metrics)
        total_successes = sum(
            m["aggregate"]["success_rate"] * m["aggregate"]["total_episodes"]
            for m in all_metrics
            if m["aggregate"]["total_episodes"] > 0
        )

        # Calculate average completion rate across all envs
        completion_rates = [m["current_episode"]["completion_rate"] for m in all_metrics]
        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0.0

        return {
            "num_envs": self.num_envs,
            "total_episodes": total_episodes,
            "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
            "avg_current_completion": avg_completion,
            "per_env_metrics": all_metrics,
        }

    def get_env_states(self) -> List[Dict[str, Any]]:
        """Get current state of all environments (useful for debugging)"""
        return [
            {
                "env_id": i,
                "is_done": env.is_done,
                "current_step": env.current_step,
                "completion_rate": env.reward_func.get_completion_rate(),
                "remaining_diffs": len(env.reward_func.get_remaining_diffs()),
                "instruction": env.instruction[:50] + "..." if len(env.instruction) > 50 else env.instruction,
            }
            for i, env in enumerate(self.envs)
        ]

    def print_status(self):
        """Print current status of all environments"""
        print(f"\n{'=' * 60}")
        print(f"MCPVecEnv Status ({self.num_envs} envs)")
        print(f"{'=' * 60}")
        for state in self.get_env_states():
            status = "DONE" if state["is_done"] else "ACTIVE"
            print(
                f"Env {state['env_id']}: [{status}] step={state['current_step']}, "
                f"completion={state['completion_rate']:.1%}, "
                f"remaining={state['remaining_diffs']}"
            )
        print(f"{'=' * 60}\n")

    async def close(self):
        """Close all environments"""
        # No persistent connections to close in current implementation
        pass


class MCPSwiftEnv(SwiftEnvBase):
    # Global counter for round-robin server selection
    _server_counter = 0
    _server_lock = None  # Will be initialized on first use

    PROMPT_TEMPLATE = load_prompt_template()

    def __init__(self, env_config=None):
        import threading

        self.env_config = env_config or {}

        # Initialize thread lock for counter if not already done
        if MCPSwiftEnv._server_lock is None:
            MCPSwiftEnv._server_lock = threading.Lock()

        # Server configuration - support multiple servers
        # Priority: MCP_SERVER_URLS env var > MCP_SERVER_URL env var > config
        default_url = "http://localhost:8080/sse"

        # Check for multiple servers first
        multi_server_env = os.environ.get("MCP_SERVER_URLS", "")
        if multi_server_env:
            # Parse comma-separated URLs
            self.server_urls = [url.strip() for url in multi_server_env.split(",") if url.strip()]
        elif "server_urls" in self.env_config:
            self.server_urls = self.env_config["server_urls"]
        else:
            # Fall back to single server
            single_url = os.environ.get("MCP_SERVER_URL", self.env_config.get("server_url", default_url))
            self.server_urls = [single_url]

        # Select server for this instance using round-robin
        with MCPSwiftEnv._server_lock:
            server_idx = MCPSwiftEnv._server_counter % len(self.server_urls)
            MCPSwiftEnv._server_counter += 1

        self.server_url = self.server_urls[server_idx]
        self.server_idx = server_idx
        self.max_steps = int(os.environ.get("MCP_MAX_STEPS", self.env_config.get("max_steps", 50)))

        print(f"[MCPSwiftEnv] Instance created: server_idx={server_idx}, url={self.server_url}")
        print(f"[MCPSwiftEnv] Available servers: {self.server_urls}")

        # Internal single environment
        self._env: Optional[MCPSingleEnv] = None
        self._initialized = False

        # Track current episode state for prompt building
        self._current_messages = []  # Full message history
        self._task = ""  # Current task instruction
        self._scene_description = ""  # Scene description
        self._task_progress = {}  # Model's self-summarized progress
        self._last_action = {}  # Last action taken
        self._last_obs = ""  # Last observation text

        # Ask tool tracking: max_ask_limit = ceil(max_steps * 0.15)
        import math

        self._max_ask_limit = math.ceil(self.max_steps * 0.15)
        self._ask_count = 0
        self._last_ask_marker = None  # marker from most recent ask response (for prompt injection)
        self._last_ask_target = None  # described target from most recent ask response

    def _get_hand_state_description(self) -> str:
        """Generate human-readable hand state for prompt."""
        if self._env and self._env.hand_occupied:
            inv_name = "an object"
            if self._env.current_inv:
                uid_part = str(self._env.current_inv)
                if "_on_" in uid_part and "_at_" in uid_part:
                    uid_part = uid_part.split("_on_")[0]
                inv_name = self._env.uid_to_category.get(uid_part, self._env.current_inv)
            return f"OCCUPIED (holding: {inv_name})"
        else:
            return "EMPTY (not holding any object)"

    def _ensure_initialized(self):
        """Lazy initialization of the environment"""
        if not self._initialized:
            self._env = MCPSingleEnv(env_id=0, server_url=self.server_url, env_config=self.env_config)
            self._initialized = True

    async def reset(self, config=None) -> Tuple[str, Dict[str, Any], str]:
        """
        Reset environment to initial state.

        Returns a single-turn prompt that matches the training format.
        NOTE: We return empty system_message and put everything in observation,
        because the model was trained with single-turn prompts, not system+user format.

        Args:
            config: RolloutInferRequest containing dataset information.

        Returns:
            Tuple of (observation, info, system_message):
            - observation: Full prompt in training format (user message)
            - info: Environment debug information
            - system_message: Empty string (prompt is self-contained)
        """
        self._ensure_initialized()

        # Extract seed from config if available
        seed = None
        if config is not None:
            if hasattr(config, "data_dict") and "seed" in config.data_dict:
                seed = config.data_dict["seed"]

        # Reset the underlying environment
        obs, info = await self._env.reset(seed=seed)

        # Store episode state
        self._task = info.get("instruction", obs)
        self._scene_description = info.get("scene_description", "")
        self._task_progress = {}
        self._last_action = {}
        self._last_obs = ""

        # Clear message history
        self._current_messages = []

        # Reset ask count for new episode
        self._ask_count = 0
        self._last_ask_marker = None
        self._last_ask_target = None

        # Add task context to info
        info["server_url"] = self.server_url
        info["max_steps"] = self.max_steps

        # Build the full prompt in training format
        # For the first step, there's no last action or observation
        observation = self.PROMPT_TEMPLATE.format(
            SCENE_DESCRIPTION=self._scene_description,
            TASK=self._task,
            TASK_PROGRESS=json.dumps(self._task_progress, indent=2) if self._task_progress else "{}",
            LAST_ACTION=json.dumps(self._last_action) if self._last_action else "{}",
            LAST_OBS="",
            max_ask_limit=self._max_ask_limit,
            ask_count=self._ask_count,
            hand_state_description=self._get_hand_state_description(),
        )

        # Return empty system message - the prompt is self-contained
        system_message = ""

        # === DEBUG: Log prompt ===
        log_file = _runtime_path("MCP_ROLLOUT_DEBUG_LOG", "logs", "rollout_debug.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            import datetime

            f.write(f"\n{'#' * 60}\n")
            f.write(f"NEW EPISODE - {datetime.datetime.now()}\n")
            f.write(f"{'#' * 60}\n")
            f.write(f"\n=== FULL PROMPT (user message) ===\n{observation}\n")
            f.write(f"\n=== INFO ===\n")
            f.write(f"Target diffs: {info.get('target_diffs')}\n")
            f.write(f"Source: {info.get('source_furniture')}\n")
            f.write(f"Destination: {info.get('destination_furniture')}\n")
            f.write(f"{'#' * 60}\n")
        print(f"\n[MCPSwiftEnv] Episode started. Debug log: {log_file}")

        return observation, info, system_message

    async def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        This method:
        1. Extracts the model response
        2. Parses the response to extract task_progress and last_action
        3. Executes the action in the underlying environment
        4. Builds a NEW full prompt (single-turn) for the next step

        Args:
            action: Either a Messages list (full dialogue history) or
                   a string/dict containing the action.
                   If Messages, the last message's content is the action.

        Returns:
            Tuple of (next_observation, reward, done, info):
            - next_observation: FULL PROMPT for next step (not just env response!)
            - reward: Reward for this step
            - done: Whether episode is finished
            - info: Debug information including reward breakdown and images
        """
        self._ensure_initialized()

        # Extract action text from various input formats
        action_text = self._extract_action_text(action)

        # === DEBUG: Log full action/response for debugging ===
        step_num = self._env.current_step + 1
        print(f"\n{'=' * 60}")
        print(f"[MCPSwiftEnv] Step {step_num} - Model Response:")
        print(f"{'=' * 60}")
        print(f"Raw action type: {type(action)}")
        if isinstance(action, list):
            print(f"Messages count: {len(action)}")
            for i, msg in enumerate(action[-3:]):  # Last 3 messages
                role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
                content = msg.get("content", str(msg))[:500] if isinstance(msg, dict) else str(msg)[:500]
                print(f"  [{i}] {role}: {content}...")
        print(f"\nExtracted action_text ({len(action_text)} chars):")
        print(f"  {action_text[:1000]}{'...' if len(action_text) > 1000 else ''}")
        print(f"{'=' * 60}\n")

        # =====================================================
        # Parse model response to extract task_progress and action
        # Format expected (from training data):
        #   <think>...</think>
        #   {"summary": {"History": "...", "New Schedule": "...", "Current subtask": "..."},
        #    "next action": {"tool_name": "...", "args": {...}}}
        # =====================================================
        self._parse_model_response(action_text)

        # Log to file for later analysis
        log_file = _runtime_path("MCP_ROLLOUT_DEBUG_LOG", "logs", "rollout_debug.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            import datetime

            f.write(f"\n{'=' * 60}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Step: {step_num}\n")
            f.write(f"Action text:\n{action_text}\n")
            f.write(f"Parsed task_progress: {json.dumps(self._task_progress)}\n")
            f.write(f"Parsed last_action: {json.dumps(self._last_action)}\n")
            f.write(f"{'=' * 60}\n")

        # Execute the action in underlying environment
        env_obs, reward, done, info = await self._env.step(action_text)

        # Track ask tool usage and sync disambiguation marker for prompt injection
        tool_name = info.get("tool_name", "")
        if tool_name == "ask":
            self._ask_count += 1
            print(f"  [Ask] Tool called, ask_count={self._ask_count}/{self._max_ask_limit}")
            if self._env and self._env._last_ask_marker:
                self._last_ask_marker = self._env._last_ask_marker
                self._last_ask_target = None
            elif self._env and self._env._last_ask_target:
                self._last_ask_target = self._env._last_ask_target
                self._last_ask_marker = None

        # Store the observation text for next prompt
        self._last_obs = env_obs

        # Add step info
        info["step"] = self._env.current_step
        info["action"] = action_text

        # =====================================================
        # Build NEXT full prompt (single-turn format matching training)
        # This is CRITICAL - the model expects single-turn prompts, not dialogue
        # =====================================================
        if not done:
            # Build observation string for the next prompt (aligned with SFT format)
            if "images" in info and info["images"]:
                # Check if there's meaningful text alongside the image (e.g., error messages)
                text_part = env_obs.replace("[Image received]", "").strip()
                if text_part:
                    obs_str = f"After performing the last action, the environment returned: {text_part}\nYou also observed the image <image>"
                else:
                    obs_str = "After performing the last action, you observed the image <image>"
            else:
                obs_str = f"After performing the last action, you observed the text: {env_obs}"

            # If the last action was ask and a specific marker was returned, remind the agent to use it
            if tool_name == "ask" and self._last_ask_marker:
                obs_str += (
                    f"\n[Disambiguation] The user specified '{self._last_ask_marker}' as the target. "
                    f"Verify it is present in the current marker map, then use it directly for focus_on and pick."
                )
            elif tool_name == "ask" and self._last_ask_target:
                obs_str += (
                    f"\n[Disambiguation] The user described the target as '{self._last_ask_target}'. "
                    f"Use the latest marker image to choose the object matching that description before focus_on and pick."
                )

            # Build the next full prompt
            next_prompt = self.PROMPT_TEMPLATE.format(
                SCENE_DESCRIPTION=self._scene_description,
                TASK=self._task,
                TASK_PROGRESS=json.dumps(self._task_progress, indent=2) if self._task_progress else "{}",
                LAST_ACTION=json.dumps(self._last_action) if self._last_action else "{}",
                LAST_OBS=obs_str,
                max_ask_limit=self._max_ask_limit,
                ask_count=self._ask_count,
                hand_state_description=self._get_hand_state_description(),
            )

            # Log the next prompt
            with open(log_file, "a") as f:
                f.write(f"\n=== NEXT PROMPT (for step {step_num + 1}) ===\n")
                f.write(f"{next_prompt[:2000]}...\n")
                f.write(f"{'=' * 60}\n")

            return next_prompt, reward, done, info
        else:
            # Episode is done, return terminal observation
            return env_obs, reward, done, info

    def _parse_model_response(self, response: str):
        """
        Parse model response to extract task_progress and last_action.

        Expected format (from training):
        <think>
        ... reasoning ...
        </think>

        {"summary": {"History": "...", "New Schedule": "...", "Current subtask": "..."},
         "next action": {"tool_name": "...", "args": {...}}}

        OR simpler format:
        {"next action": "navigate_to(desk_1)"}
        """
        # Remove think tags if present
        if "</think>" in response:
            action_part = response.split("</think>")[-1].strip()
        else:
            action_part = response.strip()

        # Try to parse JSON
        try:
            # Find the outermost JSON object
            json_match = re.search(r"\{.*\}", action_part, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))

                # Extract summary/task_progress
                if "summary" in parsed:
                    self._task_progress = parsed["summary"]
                elif "Summary" in parsed:
                    self._task_progress = parsed["Summary"]

                # Extract action
                if "next action" in parsed:
                    next_action = parsed["next action"]
                    if isinstance(next_action, dict):
                        self._last_action = next_action
                    elif isinstance(next_action, str):
                        # Preserve string shorthand like "navigate_to(desk_1)" for logging.
                        self._last_action = {"action_str": next_action}
                elif "tool_name" in parsed:
                    # Direct action format
                    self._last_action = parsed

                print(f"[MCPSwiftEnv] Parsed task_progress: {self._task_progress}")
                print(f"[MCPSwiftEnv] Parsed last_action: {self._last_action}")

        except json.JSONDecodeError as e:
            print(f"[MCPSwiftEnv] Failed to parse JSON from response: {e}")
            # Keep previous state if parsing fails
            pass

    def _extract_action_text(self, action) -> str:
        """
        Extract action text from various input formats.

        Handles:
        - Messages list (ms-swift format): Extract last assistant message
        - Dict with 'content' key
        - String directly
        - Object with .content attribute
        """
        # Messages list (ms-swift standard format)
        if isinstance(action, list):
            # Find the last assistant message
            for msg in reversed(action):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            # Fallback: last message content
            if action and isinstance(action[-1], dict):
                return action[-1].get("content", str(action[-1]))
            return str(action[-1]) if action else ""

        # Dict with content
        if isinstance(action, dict):
            return action.get("content", str(action))

        # Object with content attribute (e.g., ChatMessage)
        if hasattr(action, "content"):
            return action.content

        # Object with message.content (e.g., RolloutResponseChoice)
        if hasattr(action, "message") and hasattr(action.message, "content"):
            return action.message.content

        # String
        return str(action)

    async def close(self):
        """Clean up environment resources."""
        # No persistent connections to close
        self._initialized = False
        self._env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics (non-standard, for debugging)"""
        if self._env:
            return self._env.get_metrics()
        return {}


# Gym wrapper for compatibility (single env)
class MCPGymEnv(gym.Env):
    """Synchronous Gymnasium view of one MCP environment.

    ms-swift uses the asynchronous scheduler below, while this adapter is for
    external Gymnasium consumers. Both paths forward the reward produced by
    :class:`MCPSingleEnv`; neither path recomputes it.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config=None):
        super().__init__()
        self.env_config = env_config or {}
        server_url = self.env_config.get("server_url", "http://localhost:8000/sse")
        self._env = MCPSingleEnv(env_id=0, server_url=server_url, env_config=env_config)
        self._loop = None
        self.observation_space = gym.spaces.Text(
            max_length=int(self.env_config.get("max_observation_chars", 65536))
        )
        self.action_space = gym.spaces.Text(max_length=int(self.env_config.get("max_action_chars", 16384)))

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, coroutine):
        loop = self._get_loop()
        if loop.is_running():
            coroutine.close()
            raise RuntimeError("MCPGymEnv cannot be stepped from an active event loop; use MCPSwiftEnv instead.")
        return loop.run_until_complete(coroutine)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self._run(self._env.reset(seed=seed))
        return obs, info

    def step(self, action):
        obs, reward, terminated, info = self._run(self._env.step(action))
        numeric_reward = float(reward)
        info = dict(info)
        info["reward"] = numeric_reward
        truncated = bool(info.get("truncated", False))
        return obs, numeric_reward, bool(terminated), truncated, info

    def close(self):
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._run(self._env.disconnect())
            finally:
                self._loop.close()

    def get_metrics(self):
        return self._env.get_metrics()


# Register environments
envs["mcp"] = MCPSwiftEnv
envs["mcp_swift"] = MCPSwiftEnv
envs["mcp_vec"] = MCPSwiftEnv
envs["mcp_mock"] = MCPGymEnv


# ============================================================
# SingleTurnContextManager for single-turn prompt models
# ============================================================
# The GYMScheduler accumulates messages in multi-turn format:
#   [user1, assistant1, user2, assistant2, ...]
# But our model is trained on single-turn prompts (only one user message).
# This context manager keeps only the last user message to match training format.

try:
    from swift.plugin.context_manager import ContextManager, context_managers

    class SingleTurnContextManager(ContextManager):
        """
        Context manager that keeps only the last user message.

        This is critical for models trained on single-turn prompts.
        The MCPSwiftEnv.step() returns a FULL prompt as next_obs,
        so each user message is complete and self-contained.
        We only need the last one for the model to generate correctly.
        """

        def __init__(self, ctx_config):
            super().__init__(ctx_config)

        def manage_context(self, history, trajectory_id: str):
            """
            Keep only the last user message from history.

            Args:
                history: Full message history [user, assistant, user, assistant, ...]
                trajectory_id: Trajectory identifier (unused)

            Returns:
                Messages list with only the last user message
            """
            if not history:
                return history

            # Find the last user message
            last_user_msg = None
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg
                    break

            if last_user_msg:
                # Return only the last user message
                return [last_user_msg]
            else:
                # Fallback: return original history
                return history

    # Register the context manager
    context_managers["single_turn"] = SingleTurnContextManager
    context_managers["mcp_single_turn"] = SingleTurnContextManager
    print("[MCPSwiftEnv] Registered SingleTurnContextManager as 'single_turn' and 'mcp_single_turn'")

except ImportError as e:
    print(f"[MCPSwiftEnv] Warning: Could not register SingleTurnContextManager: {e}")
