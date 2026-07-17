"""Deterministic MCP SSE server for agent closed-loop validation.

The server exercises the public REAL tool protocol without Isaac Sim. It owns
one pick-and-place task, returns RGB observations plus structured debug state,
and records every MCP call so tests and manual model runs can prove that the
complete agent/MCP loop executed.
"""

from __future__ import annotations

import base64
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from PIL import Image, ImageDraw
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route

from mcp_server.tools import MCP_TOOLS


INITIAL_WORLD_GRAPH = {
    "source_counter": {"content": ["apple_0"]},
    "target_table": {"content": []},
}
TARGET_WORLD_GRAPH = {
    "source_counter": {"content": []},
    "target_table": {"content": ["apple_0"]},
}


def _task_info() -> dict[str, Any]:
    return {
        "task_id": "mcp-closed-loop-pick-place",
        "episode_idx": 0,
        "episode_local_idx": 0,
        "episode_range": [0, 1],
        "task_description": "Move apple_0 from source_counter to target_table.",
        "scene_id": "mcp_validation_scene",
        "rooms_and_furniture": {
            "validation_room": ["source_counter", "target_table"],
        },
        "initial_world_graph": deepcopy(INITIAL_WORLD_GRAPH),
        "target_world_graph": deepcopy(TARGET_WORLD_GRAPH),
        "target_object": {"id": "apple-0", "name": "apple_0", "category": "apple"},
        "source_furniture": "source_counter",
        "destination_furniture": "target_table",
    }


@dataclass
class ValidationState:
    task_loaded: bool = False
    completed: bool = False
    location: str | None = None
    inventory: str | None = None
    world_graph: dict[str, Any] = field(default_factory=lambda: deepcopy(INITIAL_WORLD_GRAPH))
    calls: list[dict[str, Any]] = field(default_factory=list)
    log_path: Path | None = None

    def load_task(self) -> None:
        self.task_loaded = True
        self.completed = False
        self.location = None
        self.inventory = None
        self.world_graph = deepcopy(INITIAL_WORLD_GRAPH)

    def record(self, tool_name: str, arguments: dict[str, Any], phase: str = "action") -> None:
        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": deepcopy(arguments),
                "phase": phase,
                "location": self.location,
                "inventory": self.inventory,
                "world_graph": deepcopy(self.world_graph),
            }
        )
        self.persist()

    def persist(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps(
                {
                    "task_loaded": self.task_loaded,
                    "completed": self.completed,
                    "location": self.location,
                    "inventory": self.inventory,
                    "world_graph": self.world_graph,
                    "calls": self.calls,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _text(message: str) -> types.TextContent:
    return types.TextContent(type="text", text=message)


def _observation_image(state: ValidationState, message: str) -> str:
    image = Image.new("RGB", (384, 192), "white")
    draw = ImageDraw.Draw(image)
    lines = [
        "REAL MCP validation scene",
        f"location: {state.location or 'none'}",
        f"inventory: {state.inventory or 'empty'}",
        message[:64],
    ]
    for index, line in enumerate(lines):
        draw.text((12, 12 + index * 36), line, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _action_result(state: ValidationState, message: str) -> list[types.ContentBlock]:
    debug_info = {
        "world_graph": deepcopy(state.world_graph),
        "location": state.location,
        "inventory": state.inventory,
    }
    return [
        _text(message),
        types.ImageContent(
            type="image",
            data=_observation_image(state, message),
            mimeType="image/png",
        ),
        _text("Debug Info:\n" + json.dumps(debug_info, indent=2)),
    ]


def _execute_action(
    state: ValidationState,
    tool_name: str,
    arguments: dict[str, Any],
) -> list[types.ContentBlock]:
    message: str

    if tool_name == "list_receptacles":
        message = "validation_room contains source_counter and target_table."
    elif tool_name == "navigate_to":
        destination = arguments.get("receptacle_name")
        if destination not in {"source_counter", "target_table"}:
            message = "Invalid receptacle. Choose source_counter or target_table."
        else:
            state.location = destination
            if destination == "source_counter" and state.inventory is None:
                message = "Arrived at source_counter. Use find_objects for apple."
            else:
                message = "Arrived at target_table. Use highlight_receptacles before place."
    elif tool_name in {"explore_receptacle", "find_objects", "focus_on"}:
        if state.location != "source_counter":
            message = "The apple is not visible here. Navigate to source_counter."
        elif state.inventory is not None:
            message = "apple_0 is already in inventory."
        else:
            message = "Visible object marker map: marker_id 0 is apple_0."
    elif tool_name == "highlight_receptacles":
        if state.location == "target_table":
            message = "Visible receptacle marker map: marker_id 0 is target_table surface."
        else:
            message = "Navigate to target_table before requesting its surface marker."
    elif tool_name == "pick":
        if state.location != "source_counter":
            message = "Pick failed: navigate to source_counter first."
        elif str(arguments.get("marker_id")) != "0":
            message = "Pick failed: current apple marker_id is 0."
        elif state.inventory is not None:
            message = "Pick failed: inventory is not empty."
        elif "apple_0" not in state.world_graph["source_counter"]["content"]:
            message = "Pick failed: apple_0 is no longer on source_counter."
        else:
            state.world_graph["source_counter"]["content"].remove("apple_0")
            state.inventory = "apple_0"
            message = "Pick succeeded: inventory now contains apple_0."
    elif tool_name == "place":
        if state.location != "target_table":
            message = "Place failed: navigate to target_table first."
        elif str(arguments.get("marker_id")) != "0":
            message = "Place failed: current target_table marker_id is 0."
        elif state.inventory != "apple_0":
            message = "Place failed: inventory does not contain apple_0."
        else:
            state.world_graph["target_table"]["content"].append("apple_0")
            state.inventory = None
            message = "Place succeeded: apple_0 is now on target_table."
    elif tool_name == "ask":
        message = json.dumps(
            {
                "target_category": "apple",
                "target_description": "the apple currently on source_counter",
                "source": "validation_task",
            }
        )
    elif tool_name in {"open", "close"}:
        message = f"{tool_name} is unnecessary: neither validation receptacle is articulated."
    else:
        message = f"Unknown validation action: {tool_name}."

    state.record(tool_name, arguments)
    return _action_result(state, message)


def create_validation_app(
    *,
    log_path: str | Path | None = None,
) -> tuple[Starlette, ValidationState]:
    """Create an isolated validation server and its inspectable state."""

    state = ValidationState(log_path=Path(log_path) if log_path else None)
    transport = SseServerTransport("/messages/")
    server = Server("real-agent-validation")

    @server.list_tools()
    async def list_tools(*_args) -> list[types.Tool]:
        return MCP_TOOLS

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict[str, Any]):
        arguments = arguments or {}
        if tool_name == "finish":
            if not state.task_loaded:
                state.load_task()
                state.record("finish", arguments, phase="load_task")
                return [_text(json.dumps(_task_info(), indent=2))]
            state.completed = True
            state.record("finish", arguments, phase="complete_task")
            return [_text("All evaluation episodes completed.")]
        return _execute_action(state, tool_name, arguments)

    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return Response()

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=transport.handle_post_message),
        ],
    )
    return app, state


def main() -> None:
    port = int(os.getenv("PORT", "8765"))
    host = os.getenv("HOST", "127.0.0.1")
    log_path = os.getenv("VALIDATION_LOG_PATH")
    app, _state = create_validation_app(log_path=log_path)
    print(f"REAL agent validation MCP server: http://{host}:{port}/sse")
    if log_path:
        print(f"Call log: {log_path}")
    uvicorn.run(app, host=host, port=port, log_level="info", lifespan="on")


if __name__ == "__main__":
    main()
