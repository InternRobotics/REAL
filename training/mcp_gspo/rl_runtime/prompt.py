"""Loading and validation for the model-facing REAL rollout prompt."""

from __future__ import annotations

import os
import string
from pathlib import Path

from .real_contract import REAL_ACTION_TOOLS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "real_grpo_prompt.txt"
REQUIRED_FIELDS = {
    "SCENE_DESCRIPTION",
    "TASK",
    "TASK_PROGRESS",
    "LAST_ACTION",
    "LAST_OBS",
    "max_ask_limit",
    "ask_count",
    "hand_state_description",
}
LEGACY_TOOL_NAMES = {
    "nav_to",
    "walk_around",
    "gaze_at",
    "show_object_by_category",
    "show_receptacles",
}
REQUIRED_TOOL_NAMES = (*REAL_ACTION_TOOLS, "finish", "ask")


class PromptValidationError(ValueError):
    """Raised when a synchronized prompt is incompatible with the runtime."""


def validate_prompt_template(template: str) -> None:
    if not template.strip():
        raise PromptValidationError("prompt is empty")

    fields = {field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name is not None}
    missing_fields = REQUIRED_FIELDS - fields
    if missing_fields:
        raise PromptValidationError(f"prompt is missing required fields: {sorted(missing_fields)}")

    missing_tools = {name for name in REQUIRED_TOOL_NAMES if f'"name": "{name}"' not in template}
    if missing_tools:
        raise PromptValidationError(f"prompt is missing canonical REAL tools: {sorted(missing_tools)}")

    legacy_tools = {name for name in LEGACY_TOOL_NAMES if f'"{name}"' in template}
    if legacy_tools:
        raise PromptValidationError(f"prompt advertises legacy tool names: {sorted(legacy_tools)}")


def load_prompt_template(path: str | os.PathLike[str] | None = None) -> str:
    prompt_path = Path(path or os.environ.get("MCP_PROMPT_PATH", DEFAULT_PROMPT_PATH))
    try:
        template = prompt_path.read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        raise PromptValidationError(f"cannot read prompt at {prompt_path}: {exc}") from exc
    validate_prompt_template(template)
    return template
