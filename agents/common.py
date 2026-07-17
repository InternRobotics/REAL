"""Shared MCP protocol and trajectory handling for REAL agents."""

from __future__ import annotations

import ast
import base64
import json
import re
import time
import traceback
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse

from PIL import Image


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_ALL_EPISODES_COMPLETED = "All evaluation episodes completed."


@dataclass(frozen=True)
class AgentRunConfig:
    """Configuration shared by both agent backends."""

    model_name: str
    mcp_server_url: str
    output_base: Path = Path("eval_output")
    max_steps: int = 20
    start_idx: int = 0
    end_idx: int = 999_999
    result_prefix: str = "agent"
    run_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.mcp_server_url:
            raise ValueError("mcp_server_url must not be empty")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.start_idx < 0 or self.end_idx < self.start_idx:
            raise ValueError("episode range must satisfy 0 <= start_idx <= end_idx")

    @property
    def model_slug(self) -> str:
        return safe_path_component(self.model_name.replace("-", "_").replace("/", "_"))


@dataclass(frozen=True)
class ActionDecision:
    """One normalized action emitted by a policy backend."""

    tool_name: Optional[str]
    arguments: dict[str, Any]
    raw_response: Any
    tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolObservation:
    """Normalized content returned by an MCP tool call."""

    result_type: str
    data: str
    debug_info: dict[str, Any]
    images_b64: list[str]
    image_paths: list[Path] = field(default_factory=list)


class AgentPolicy(Protocol):
    async def decide(self, step_dir: Path) -> ActionDecision:
        """Return one action decision for the current episode state."""

    def observe(self, decision: ActionDecision, observation: ToolObservation) -> None:
        """Update policy state after a non-finish action."""


PolicyFactory = Callable[[dict[str, Any], Sequence[Any], Path], AgentPolicy]


def _loopback_httpx_client_factory(url: str):
    """Bypass ambient proxies only when MCP is explicitly on loopback."""

    hostname = urlparse(url).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None

    def create_client(headers=None, timeout=None, auth=None):
        import httpx

        kwargs = {
            "follow_redirects": True,
            "trust_env": False,
        }
        if headers is not None:
            kwargs["headers"] = headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        if auth is not None:
            kwargs["auth"] = auth
        return httpx.AsyncClient(**kwargs)

    return create_client


def safe_path_component(value: Any) -> str:
    """Turn server/model metadata into one safe trajectory path component."""

    cleaned = _UNSAFE_PATH_CHARS.sub("_", str(value)).strip("._")
    return cleaned or "unknown"


def episode_output_dir(config: AgentRunConfig, task_info: Mapping[str, Any]) -> Path:
    task_identity = task_info.get("benchmark_task_id")
    if not task_identity and task_info.get("family"):
        task_identity = f"{task_info['family']}/{task_info.get('task_id', 'unknown_task')}"
    if not task_identity:
        task_identity = task_info.get("task_id", "unknown_task")
    return (
        config.output_base
        / safe_path_component(task_info.get("scene_id", "unknown_scene"))
        / safe_path_component(task_identity)
        / config.model_slug
    )


def build_scene_description(rooms_and_furniture: Mapping[str, Sequence[str]]) -> str:
    lines = []
    for index, (room, furniture) in enumerate(sorted(rooms_and_furniture.items()), 1):
        room_name = str(room).replace("/", "_").replace(" ", "_")
        lines.append(f"{index}) In {room_name}: {', '.join(sorted(furniture))}.")
    return "\n".join(lines)


def mcp_tool_to_openai(mcp_tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool into an OpenAI-compatible function schema."""

    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        },
    }


def format_mcp_tools_for_prompt(mcp_tools: Sequence[Any]) -> str:
    """Render the server's live tool schemas for text-only action generation."""

    lines = []
    for tool in mcp_tools:
        schema = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }
        lines.append(f"- {json.dumps(schema, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


def evaluate_world_graph(actual: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """Compare object contents and articulation state represented in the goal graph."""

    if not isinstance(actual, Mapping) or not isinstance(target, Mapping) or not target:
        return False
    for furniture, target_state in target.items():
        actual_state = actual.get(furniture)
        if not isinstance(target_state, Mapping) or not isinstance(actual_state, Mapping):
            return False
        if "content" in target_state:
            target_content = target_state["content"]
            actual_content = actual_state.get("content")
            if not isinstance(target_content, list) or not isinstance(actual_content, list):
                return False
            if sorted(target_content) != sorted(actual_content):
                return False
        if "door" in target_state and actual_state.get("door") != target_state["door"]:
            return False
    return True


def parse_tool_result(result: Any) -> ToolObservation:
    """Split user-facing text/images from the server's debug payload."""

    texts: list[str] = []
    images_b64: list[str] = []
    debug_info: dict[str, Any] = {}

    for content in getattr(result, "content", []):
        content_type = getattr(content, "type", "")
        if content_type == "text":
            text = content.text
            if text.startswith("Debug Info:\n"):
                raw_debug = text.removeprefix("Debug Info:\n")
                try:
                    parsed_debug = json.loads(raw_debug)
                    debug_info = (
                        parsed_debug if isinstance(parsed_debug, dict) else {"raw": raw_debug}
                    )
                except (TypeError, json.JSONDecodeError):
                    debug_info = {"raw": raw_debug}
            else:
                texts.append(text)
        elif content_type == "image":
            images_b64.append(content.data)

    if images_b64 and not texts:
        return ToolObservation("image", images_b64[0], debug_info, images_b64)
    return ToolObservation("text", "\n".join(texts).strip(), debug_info, images_b64)


def extract_world_graph_from_debug(debug_info: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Read the current graph from current or legacy debug payloads."""

    if not isinstance(debug_info, Mapping):
        return None

    world_graph = debug_info.get("world_graph")
    if isinstance(world_graph, dict):
        return world_graph

    legacy_graph = debug_info.get("WORLD_GRAPH")
    if isinstance(legacy_graph, dict):
        return legacy_graph
    if not isinstance(legacy_graph, str):
        return None

    parsed: dict[str, Any] = {}
    for raw_line in legacy_graph.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        furniture, content_text = line.split(":", 1)
        try:
            content = ast.literal_eval(content_text.strip())
        except (SyntaxError, ValueError):
            content = []
        parsed[furniture.strip()] = {"content": content if isinstance(content, list) else []}
    return parsed or None


async def fetch_next_task(session: Any) -> Optional[dict[str, Any]]:
    """Finish the current server slot and return the next task, if any."""

    result = await session.call_tool("finish", arguments={})
    for content in getattr(result, "content", []):
        if getattr(content, "type", None) != "text":
            continue
        text = content.text.strip()
        if text == _ALL_EPISODES_COMPLETED:
            return None
        try:
            task_info = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"finish returned invalid task metadata: {text[:200]}") from error
        if not isinstance(task_info, dict):
            raise RuntimeError("finish returned task metadata that is not an object")
        return task_info
    raise RuntimeError("finish returned no textual task metadata")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_raw_response(step_dir: Path, raw_response: Any) -> None:
    if isinstance(raw_response, str):
        (step_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    else:
        _write_json(step_dir / "raw_response.json", raw_response)


def _decode_base64_image(image_b64: str, out_path: Path) -> None:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    image.save(out_path)


def _save_step_artifacts(
    step_dir: Path,
    decision: ActionDecision,
    observation: ToolObservation,
    inference_time: float,
    total_inference_time: float,
) -> ToolObservation:
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        step_dir / "tool_call.json",
        {"tool_name": decision.tool_name, "args": decision.arguments},
    )
    _write_json(
        step_dir / "inference_time.json",
        {
            "inference_time": inference_time,
            "total_inference_time": total_inference_time,
        },
    )
    _write_raw_response(step_dir, decision.raw_response)

    if observation.result_type == "text":
        (step_dir / "textual_observation.txt").write_text(
            observation.data,
            encoding="utf-8",
        )

    image_paths = []
    for index, image_b64 in enumerate(observation.images_b64):
        image_path = step_dir / f"image_observation_{index}.png"
        _decode_base64_image(image_b64, image_path)
        image_paths.append(image_path)

    _write_json(step_dir / "debug_info.json", observation.debug_info)
    return replace(observation, image_paths=image_paths)


def _write_task_metadata(
    trajectory_dir: Path,
    task_info: Mapping[str, Any],
    config: AgentRunConfig,
) -> None:
    fields = (
        "benchmark_task_id",
        "family",
        "global_index",
        "task_id",
        "episode_idx",
        "task_description",
        "scene_id",
        "source_furniture",
        "destination_furniture",
        "target_object",
    )
    metadata = {field: task_info.get(field) for field in fields}
    metadata.update({"model": config.model_name, **config.run_metadata})
    _write_json(trajectory_dir / "task_meta.json", metadata)


async def run_episode(
    session: Any,
    task_info: dict[str, Any],
    config: AgentRunConfig,
    policy: AgentPolicy,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Run and score one task using an already initialized policy."""

    task_id = str(task_info["task_id"])
    episode_idx = int(task_info.get("episode_idx", -1))
    trajectory_dir = episode_output_dir(config, task_info)

    if trajectory_dir.exists() and any(trajectory_dir.iterdir()):
        print(f"[SKIP] Episode {episode_idx} ({task_id}) already has output.")
        return {
            "benchmark_task_id": task_info.get("benchmark_task_id"),
            "family": task_info.get("family"),
            "global_index": task_info.get("global_index"),
            "task_id": task_id,
            "episode_idx": episode_idx,
            "status": "skipped",
        }, (await fetch_next_task(session))

    trajectory_dir.mkdir(parents=True, exist_ok=True)
    _write_task_metadata(trajectory_dir, task_info, config)

    latest_world_graph = task_info.get("initial_world_graph", {})
    total_inference_time = 0.0
    steps_taken = 0
    final_action: Optional[str] = None
    success = False
    error_kind: Optional[str] = None
    next_task: Optional[dict[str, Any]] = None

    for step_index in range(config.max_steps):
        step_dir = trajectory_dir / f"step_{step_index:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        try:
            decision = await policy.decide(step_dir)
        except Exception:
            (step_dir / "inference_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            error_kind = "inference_error"
            steps_taken = step_index + 1
            break

        inference_time = time.perf_counter() - start_time
        total_inference_time += inference_time
        steps_taken = step_index + 1

        if not decision.tool_name:
            empty_observation = ToolObservation("text", "No tool selected.", {}, [])
            _save_step_artifacts(
                step_dir,
                decision,
                empty_observation,
                inference_time,
                total_inference_time,
            )
            continue

        print(
            f"  Step {step_index}: {decision.tool_name}({decision.arguments}) "
            f"[{inference_time:.1f}s]"
        )

        if decision.tool_name == "finish":
            success = evaluate_world_graph(
                latest_world_graph,
                task_info.get("target_world_graph", {}),
            )
            finish_observation = ToolObservation(
                "text",
                f"Episode finished. Success: {success}",
                {"world_graph": latest_world_graph},
                [],
            )
            _save_step_artifacts(
                step_dir,
                decision,
                finish_observation,
                inference_time,
                total_inference_time,
            )
            final_action = "finish"
            next_task = await fetch_next_task(session)
            break

        try:
            result = await session.call_tool(decision.tool_name, arguments=decision.arguments)
            observation = parse_tool_result(result)
            observation = _save_step_artifacts(
                step_dir,
                decision,
                observation,
                inference_time,
                total_inference_time,
            )
            policy.observe(decision, observation)
        except Exception:
            (step_dir / "action_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            error_kind = "action_error"
            break

        parsed_world_graph = extract_world_graph_from_debug(observation.debug_info)
        if parsed_world_graph is not None:
            latest_world_graph = parsed_world_graph

    if final_action is None:
        next_task = await fetch_next_task(session)

    status = "finished" if final_action == "finish" else error_kind or "max_steps_reached"
    summary = {
        "benchmark_task_id": task_info.get("benchmark_task_id"),
        "family": task_info.get("family"),
        "global_index": task_info.get("global_index"),
        "task_id": task_id,
        "episode_idx": episode_idx,
        "total_steps": steps_taken,
        "total_inference_time": total_inference_time,
        "final_action": final_action,
        "success": success,
        "status": status,
    }
    _write_json(trajectory_dir / "episode_summary.json", summary)
    print(f"  Episode {episode_idx} ({task_id}): {status} success={success} in {steps_taken} steps")
    return summary, next_task


def _summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finished = sum(result.get("status") == "finished" for result in results)
    successes = sum(bool(result.get("success")) for result in results)
    skipped = sum(result.get("status") == "skipped" for result in results)
    errors = sum(str(result.get("status", "")).endswith("_error") for result in results)
    return {
        "total": len(results),
        "finished": finished,
        "successes": successes,
        "skipped": skipped,
        "errors": errors,
        "success_rate": successes / finished if finished else 0.0,
    }


async def run_evaluation(config: AgentRunConfig, policy_factory: PolicyFactory) -> dict[str, Any]:
    """Connect to an MCP server and evaluate every task in the configured range."""

    from mcp import ClientSession
    from mcp.client.sse import sse_client

    config.output_base.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    transport_options = {}
    direct_client_factory = _loopback_httpx_client_factory(config.mcp_server_url)
    if direct_client_factory is not None:
        transport_options["httpx_client_factory"] = direct_client_factory

    async with sse_client(config.mcp_server_url, **transport_options) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed_tools = await session.list_tools()
            tools = listed_tools.tools
            tool_names = {tool.name for tool in tools}
            if "finish" not in tool_names:
                raise RuntimeError("MCP server does not expose the required finish tool")
            print(f"Connected to MCP server, tools={len(tools)}")

            task_info = await fetch_next_task(session)
            while task_info is not None:
                episode_idx = int(task_info.get("episode_idx", -1))
                if episode_idx < config.start_idx:
                    task_info = await fetch_next_task(session)
                    continue
                if episode_idx >= config.end_idx:
                    break

                print(f"\n[Episode {episode_idx}] task_id={task_info.get('task_id')}")
                policy = policy_factory(
                    task_info,
                    tools,
                    episode_output_dir(config, task_info),
                )
                result, task_info = await run_episode(session, task_info, config, policy)
                results.append(result)

    counts = _summarize_results(results)
    run_tag = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    run_summary = {
        "mode": config.result_prefix,
        "run_tag": run_tag,
        "model": config.model_name,
        "start_idx": config.start_idx,
        "end_idx": config.end_idx,
        **config.run_metadata,
        **counts,
        "episodes": results,
    }
    result_path = (
        config.output_base
        / f"{safe_path_component(config.result_prefix)}_results_{config.model_slug}_{run_tag}.json"
    )
    _write_json(result_path, run_summary)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete: {counts['total']} episodes")
    print(f"  Finished:  {counts['finished']}")
    print(f"  Successes: {counts['successes']}")
    print(f"  Skipped:   {counts['skipped']}")
    print(f"  Errors:    {counts['errors']}")
    if counts["finished"]:
        print(f"  Success rate: {counts['success_rate'] * 100:.1f}%")
    print(f"  Results: {result_path}")
    print(f"{'=' * 60}")
    return run_summary
