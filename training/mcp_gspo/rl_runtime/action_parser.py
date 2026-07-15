"""Strict parsing for model-generated REAL tool actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .real_contract import MODEL_ACTION_SCHEMAS, canonical_tool_name


class ActionParseError(ValueError):
    """Raised when a completion does not contain a usable tool action."""


@dataclass(frozen=True)
class ParsedAction:
    requested_tool_name: str
    tool_name: str
    arguments: dict[str, Any]


def _json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _parse_shorthand(value: str) -> tuple[str, dict[str, Any]]:
    match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*", value, re.DOTALL)
    if not match:
        raise ActionParseError(f"invalid action shorthand: {value!r}")

    requested_name = match.group(1)
    canonical_name = canonical_tool_name(requested_name)
    raw_argument = (match.group(2) or "").strip()
    if not raw_argument:
        return requested_name, {}

    schema = MODEL_ACTION_SCHEMAS.get(canonical_name, {})
    required = list(schema.get("required", []))
    if len(required) != 1:
        raise ActionParseError(f"{requested_name} shorthand is ambiguous; emit JSON arguments")

    try:
        argument = json.loads(raw_argument)
    except json.JSONDecodeError:
        argument = raw_argument.strip("\"'")
    return requested_name, {required[0]: argument}


def _from_object(value: dict[str, Any]) -> ParsedAction:
    action: Any = value.get("next action", value.get("next_action", value))
    if isinstance(action, str):
        requested_name, arguments = _parse_shorthand(action)
    elif isinstance(action, dict):
        requested_name = action.get("tool_name") or action.get("tool") or action.get("name")
        arguments = action.get("args", action.get("arguments", {}))
    else:
        raise ActionParseError("action must be an object or function shorthand")

    if not isinstance(requested_name, str) or not requested_name.strip():
        raise ActionParseError("tool_name is missing")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ActionParseError("args must be a JSON object")

    requested_name = requested_name.strip()
    return ParsedAction(
        requested_tool_name=requested_name,
        tool_name=canonical_tool_name(requested_name),
        arguments=dict(arguments),
    )


def parse_action(action_text: str) -> ParsedAction:
    """Parse the first valid action object from a model completion."""
    if not isinstance(action_text, str) or not action_text.strip():
        raise ActionParseError("completion is empty")

    candidate_text = action_text.rsplit("</think>", 1)[-1]
    errors: list[str] = []
    for value in _json_objects(candidate_text):
        try:
            return _from_object(value)
        except ActionParseError as exc:
            errors.append(str(exc))

    detail = f": {'; '.join(errors)}" if errors else ""
    raise ActionParseError(f"no valid JSON tool action found{detail}")
