"""
MCP evaluation server — non-headless GUI mode.

mcp_env_demo is aliased as mcp_server.mcp_env in sys.modules so that
actions.py shares the already-running simulation instance.

Usage:
    DEMO_TASK_CONFIG=configs/demo_task.yaml \\
    TRAJ_PATH=eval_output_demo \\
    PORT=8080 \\
    python -m mcp_server.mcp_server_demo

Or simply:
    ./run_mcp_server_demo.sh
"""

import asyncio
import base64
import json
import os
from io import BytesIO
from copy import deepcopy
from typing import List
from dataclasses import dataclass

from PIL import Image
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route
import uvicorn

os.environ.setdefault("IS_TEST", "0")

# Alias mcp_env_demo as mcp_server.mcp_env so actions.py shares this instance.
import sys
import mcp_server.mcp_env_demo as _mcp_env_module

sys.modules["mcp_server.mcp_env"] = _mcp_env_module

from mcp_server.mcp_env_demo import (  # noqa: E402
    env,
    camera,
    processed_eval_episodes,
    spawn_objects_by_world_graph,
    TARGET_SCENE_ID,
    object_per_room,
)
from mcp_server.perception_utils import init_annotators, get_rgb_image  # noqa: E402
from mcp_server.tools import MCP_TOOLS  # noqa: E402
from mcp_server.actions import (  # noqa: E402
    EvalState,
    dispatch_action,
    step_simulation,
    _ensure_playing,
)


# =============================================================================
# MCP Server Setup
# =============================================================================

sse = SseServerTransport("/messages/")
app = Server("eval-server-demo")


@app.list_tools()
async def list_tools(*args) -> list[types.Tool]:
    return MCP_TOOLS


# =============================================================================
# Task Manager
# =============================================================================


@dataclass
class Task:
    action_name: str
    action_args: dict
    future: asyncio.Future


class TaskManager:
    def __init__(self):
        self.task = None

    def register(self, action: str, args: dict) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self.task = Task(action_name=action, action_args=args, future=future)
        return future

    def return_result(self, result: list):
        if self.task is not None:
            future = self.task.future
            self.task = None
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, result)

    def has_task(self) -> bool:
        return self.task is not None


# =============================================================================
# Global State
# =============================================================================

state = EvalState()
manager = TaskManager()
current_episode_idx = -1

EPISODE_START_IDX = int(os.getenv("START_IDX", "0"))
EPISODE_END_IDX = int(os.getenv("END_IDX", str(len(processed_eval_episodes))))
EPISODE_END_IDX = min(EPISODE_END_IDX, len(processed_eval_episodes))

if EPISODE_START_IDX < 0:
    EPISODE_START_IDX = 0
if EPISODE_START_IDX > EPISODE_END_IDX:
    EPISODE_START_IDX = EPISODE_END_IDX

eval_episodes = processed_eval_episodes[EPISODE_START_IDX:EPISODE_END_IDX]

init_annotators(camera, resolution=(640, 480))


# =============================================================================
# MCP Handler
# =============================================================================


@app.call_tool()
async def wrapped_handler(
    action_name: str, arguments: dict
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    future = manager.register(action_name, arguments)
    result = await future
    return result


# =============================================================================
# HTTP Routes
# =============================================================================


async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())
    return Response()


starlette_app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)


# =============================================================================
# Action Handlers
# =============================================================================


def handle_finish() -> list:
    global current_episode_idx, state

    current_episode_idx += 1

    if current_episode_idx >= len(eval_episodes):
        return [types.TextContent(type="text", text="All evaluation episodes completed.")]

    episode = eval_episodes[current_episode_idx]
    global_episode_idx = EPISODE_START_IDX + current_episode_idx

    try:
        state.current_extra_assets = spawn_objects_by_world_graph(
            env, episode, state.current_extra_assets
        )
    except Exception:
        import traceback

        return [
            types.TextContent(
                type="text",
                text=f"Error spawning episode {current_episode_idx}: {traceback.format_exc()}",
            )
        ]

    state.world_graph = deepcopy(episode["initial_world_graph"])
    step_simulation(env, 500)

    state.current_obs_dict = None
    state.current_marker_map = None
    state.current_landmark = None
    state.current_inv = None
    state.current_pos = None
    state.camera_orientation = None
    state.simulated_user_context = deepcopy(episode.get("simulated_user_context", {}))

    target_world_graph = deepcopy(episode["initial_world_graph"])
    src = episode["src"]
    dest = episode["dest"]
    target_obj_name = episode["target_object_name"]

    if src in target_world_graph and target_obj_name in target_world_graph[src]["content"]:
        target_world_graph[src]["content"].remove(target_obj_name)
    if dest in target_world_graph:
        if target_obj_name not in target_world_graph[dest]["content"]:
            target_world_graph[dest]["content"].append(target_obj_name)

    task_info = {
        "task_id": episode["task_id"],
        **{
            field: episode[field]
            for field in ("benchmark_task_id", "family", "global_index")
            if field in episode
        },
        "episode_idx": global_episode_idx,
        "episode_local_idx": current_episode_idx,
        "episode_range": [EPISODE_START_IDX, EPISODE_END_IDX],
        "task_description": episode["task_description"],
        "scene_id": TARGET_SCENE_ID,
        "rooms_and_furniture": dict(object_per_room),
        "initial_world_graph": episode["initial_world_graph"],
        "target_world_graph": target_world_graph,
        "target_object": {
            "id": episode["target_object_id"],
            "name": episode["target_object_name"],
            "category": episode["target_category"],
        },
        "source_furniture": src,
        "destination_furniture": dest,
        "source_distractors": episode.get("src_distractors", []),
        "destination_distractors": episode.get("dest_distractors", []),
        "object_distractors": episode.get("obj_distractors", []),
        "distractor_metadata": episode.get("obj_distractor_meta", {}),
        "execution_plan": episode.get("execution_plan", []),
    }

    return [types.TextContent(type="text", text=json.dumps(task_info, indent=2))]


def execute_action(action_name: str, arguments: dict) -> list:
    global state

    if action_name == "finish":
        return handle_finish()

    result_type, result_data, debug_info = dispatch_action(action_name, arguments, state, env)

    step_simulation(env, 200)

    result = []

    if result_type == "text":
        result.append(types.TextContent(type="text", text=result_data))
    elif result_type == "image":
        result.append(types.ImageContent(type="image", data=result_data, mimeType="image/png"))

    if result_type != "image":
        rgb = Image.fromarray(get_rgb_image())
        buffered = BytesIO()
        rgb.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        result.append(types.ImageContent(type="image", data=img_str, mimeType="image/png"))

    debug_str = json.dumps(debug_info, indent=2, default=str)
    result.append(types.TextContent(type="text", text=f"Debug Info:\n{debug_str}"))

    return result


# =============================================================================
# Main Loop
# =============================================================================


def run_api():
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "127.0.0.1")
    config = uvicorn.Config(starlette_app, host=host, port=port, lifespan="on")
    server = uvicorn.Server(config)
    server.run()


def main():
    import threading

    api_thread = threading.Thread(target=run_api, name="HTTP-Thread", daemon=True)
    api_thread.start()

    print(
        f"[DEMO] MCP Server running at "
        f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', 8080)}/sse"
    )
    print(f"[DEMO] Scene: {TARGET_SCENE_ID}  (non-headless GUI mode)")
    print(f"[DEMO] Episodes: {len(eval_episodes)} (range [{EPISODE_START_IDX}, {EPISODE_END_IDX}))")

    # Agent clients obtain their first task by calling `finish`, so preloading
    # must be opt-in or that first call would skip the preloaded episode.  The
    # preview mode remains available for standalone visual inspection.
    auto_load_episode = os.getenv("AUTO_LOAD_EPISODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if eval_episodes and auto_load_episode:
        global current_episode_idx
        demo_local_idx = next(
            (i for i, ep in enumerate(eval_episodes) if len(ep["placements"]) >= 2),
            0,
        )
        current_episode_idx = demo_local_idx - 1  # handle_finish() increments by 1

        print(
            f"[DEMO] Auto-loading episode local_idx={demo_local_idx} "
            f"({len(eval_episodes[demo_local_idx]['placements'])} objects)..."
        )
        init_result = handle_finish()
        print(f"[DEMO] Spawn done: {init_result[0].text[:200] if init_result else 'none'}")
    elif eval_episodes:
        print("[DEMO] Waiting for an agent to request the first episode...")

    while env.simulation_app.is_running():
        _ensure_playing()

        if not manager.has_task():
            env.step([{}])
            continue

        action_name = manager.task.action_name
        arguments = manager.task.action_args

        try:
            result = execute_action(action_name, arguments)
            manager.return_result(result)
        except Exception:
            import traceback

            error_msg = f"Error executing {action_name}: {traceback.format_exc()}"
            print(error_msg)
            manager.return_result([types.TextContent(type="text", text=error_msg)])

    env.simulation_app.close()


if __name__ == "__main__":
    main()
