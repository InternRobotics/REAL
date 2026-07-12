"""Helpers for loading REAL task configuration files."""

import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "demo_task.yaml"


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def load_task_config(path: str | Path | None = None) -> dict:
    """Load a task config without requiring machine-specific files."""
    config_path = resolve_repo_path(
        path or os.environ.get("DEMO_TASK_CONFIG", DEFAULT_CONFIG_PATH)
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if "scene_id" not in config or "paths" not in config:
        raise ValueError(
            f"Task config must define 'scene_id' and 'paths': {config_path}"
        )
    return config


def get_metadata_path(filename: str) -> Path:
    """Path to a bundled metadata file (in repo metadata/ dir)."""
    return _PROJECT_ROOT / "metadata" / filename
