"""Deterministic simulated-user context for interactive MCP episodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_simulated_user_context(
    episode: Mapping[str, Any],
    fallback_category: str | None = None,
) -> dict[str, str]:
    """Extract the canonical interactive target from task metadata.

    Task-generator JSON plans contain a structured ``ask`` step, while exported
    YAML plans contain ``"ask <category> - <description>"``.  Supporting both
    representations keeps the response deterministic and independent of a live
    human or of arguments supplied by the MCP client.
    """
    target_category = ""
    target_description = ""

    for step in episode.get("execution_plan", []):
        if isinstance(step, Mapping) and step.get("action") == "ask":
            args = step.get("args", {})
            if isinstance(args, Mapping):
                target_category = str(args.get("target_category", "")).strip()
                target_description = str(args.get("target_description", "")).strip()
        elif isinstance(step, str) and step.strip().lower().startswith("ask "):
            payload = step.strip()[4:]
            category, separator, description = payload.partition(" - ")
            if separator:
                target_category = category.strip()
                target_description = description.strip()

        if target_description:
            break

    if not target_description:
        target_description = str(episode.get("target_description", "")).strip()
    if not target_category:
        target_category = str(episode.get("target_category") or fallback_category or "").strip()

    if not target_description:
        return {}

    return {
        "target_category": target_category,
        "target_description": target_description,
        "source": "task_metadata",
    }
