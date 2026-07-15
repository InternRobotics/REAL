"""REAL MCP action contract and compatibility helpers.

The canonical schemas mirror ``mcp_server/tools.py`` in the REAL repository.
Training-only lifecycle tools are intentionally kept separate because the public
demo server does not advertise them through ``list_tools``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

REAL_UPSTREAM_URL = "git@github.com:InternRobotics/REAL.git"
REAL_UPSTREAM_COMMIT = "e26838d723e3831a53dcabf2e16ba494a5aa1c25"

REAL_ACTION_TOOLS: tuple[str, ...] = (
    "list_receptacles",
    "navigate_to",
    "explore_receptacle",
    "focus_on",
    "find_objects",
    "highlight_receptacles",
    "pick",
    "place",
    "open",
    "close",
)

# These are supplied by the training/evaluation server rather than REAL's public
# action-tool list. ``finish`` is also used client-side to terminate a rollout.
TRAINING_CONTROL_TOOLS: tuple[str, ...] = (
    "finish",
    "finish_with_id",
    "list_tasks",
    "ask",
)

REAL_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_receptacles": {"required": [], "properties": {}},
    "navigate_to": {
        "required": ["receptacle_name"],
        "properties": {"receptacle_name": {"type": "string"}},
    },
    "explore_receptacle": {"required": [], "properties": {}},
    "focus_on": {
        "required": ["marker_id"],
        "properties": {"marker_id": {"type": "string"}},
    },
    "find_objects": {
        "required": ["target_category"],
        "properties": {"target_category": {"type": "string"}},
    },
    "highlight_receptacles": {"required": [], "properties": {}},
    "pick": {
        "required": ["marker_id"],
        "properties": {"marker_id": {"type": "string"}},
    },
    "place": {
        "required": ["marker_id"],
        "properties": {"marker_id": {"type": "string"}},
    },
    "open": {
        "required": ["marker_id"],
        "properties": {"marker_id": {"type": "string"}},
    },
    "close": {
        "required": ["marker_id"],
        "properties": {"marker_id": {"type": "string"}},
    },
}

MODEL_CONTROL_SCHEMAS: dict[str, dict[str, Any]] = {
    "finish": {"required": [], "properties": {}},
    "ask": {
        "required": ["question"],
        "properties": {"question": {"type": "string"}},
    },
}

MODEL_ACTION_SCHEMAS = {**REAL_TOOL_SCHEMAS, **MODEL_CONTROL_SCHEMAS}

LEGACY_TOOL_ALIASES: dict[str, str] = {
    "nav_to": "navigate_to",
    "walk_around": "explore_receptacle",
    "gaze_at": "focus_on",
    "show_object_by_category": "find_objects",
    "show_receptacles": "highlight_receptacles",
    "place_on_top": "place",
}


def canonical_tool_name(tool_name: str) -> str:
    """Return the REAL name for a canonical or legacy action name."""
    normalized = str(tool_name or "").strip()
    return LEGACY_TOOL_ALIASES.get(normalized, normalized)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _tool_schema(tool: Any) -> Mapping[str, Any]:
    if isinstance(tool, Mapping):
        return tool.get("inputSchema", tool.get("input_schema", {})) or {}
    return getattr(tool, "inputSchema", getattr(tool, "input_schema", {})) or {}


def resolve_server_tool(tool_name: str, discovered_tools: Iterable[Any]) -> str:
    """Resolve an action to the name exposed by the connected MCP server.

    Canonical REAL names win. Legacy names are accepted only as a compatibility
    fallback for older private training servers.
    """
    requested = str(tool_name or "").strip()
    canonical = canonical_tool_name(requested)
    names = {_tool_name(tool) for tool in discovered_tools}
    if canonical in names:
        return canonical
    if requested in names:
        return requested
    for legacy, target in LEGACY_TOOL_ALIASES.items():
        if target == canonical and legacy in names:
            return legacy
    return canonical


def normalize_server_arguments(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    discovered_tools: Iterable[Any],
) -> dict[str, Any]:
    """Adapt a single legacy argument key to the discovered server schema."""
    normalized = dict(arguments or {})
    server_name = resolve_server_tool(tool_name, discovered_tools)
    tool = next((item for item in discovered_tools if _tool_name(item) == server_name), None)
    if tool is None:
        return normalized

    required = list(_tool_schema(tool).get("required", []))
    missing = [key for key in required if key not in normalized]
    if len(required) == 1 and len(missing) == 1 and len(normalized) == 1:
        # This covers the old open/close ``receptacle_name`` schema and similar
        # one-argument aliases without guessing when multiple values are present.
        normalized = {required[0]: next(iter(normalized.values()))}
    return normalized


def validate_real_action_tools(discovered_tools: Iterable[Any]) -> list[str]:
    """Return actionable errors for deviations from the public REAL contract."""
    tools_by_name = {_tool_name(tool): tool for tool in discovered_tools}
    errors: list[str] = []
    for name in REAL_ACTION_TOOLS:
        tool = tools_by_name.get(name)
        if tool is None:
            errors.append(f"missing REAL action tool: {name}")
            continue
        schema = _tool_schema(tool)
        expected_required = set(REAL_TOOL_SCHEMAS[name]["required"])
        actual_required = set(schema.get("required", []))
        if actual_required != expected_required:
            errors.append(f"{name} required args {sorted(actual_required)} != {sorted(expected_required)}")
    return errors
