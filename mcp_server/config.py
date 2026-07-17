"""Helpers for loading REAL task configuration files."""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "demo_task.yaml"
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_REQUIRED_PATHS = ("scene_usd", "occ_map_dir", "furniture_lib")
_REQUIRED_OBJECT_FIELDS = ("original_id", "category", "usd_path", "usd_scale", "position")


def _is_xyz(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def validate_task_config(config: Any, source: str | Path = "<task config>") -> dict:
    """Validate the runtime schema consumed by the eval server."""
    label = str(source)
    if not isinstance(config, dict):
        raise ValueError(f"Task config must contain a mapping: {label}")

    scene_id = config.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError(f"Task config must define a non-empty 'scene_id': {label}")

    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError(f"Task config must define a 'paths' mapping: {label}")
    for field in _REQUIRED_PATHS:
        if not isinstance(paths.get(field), str) or not paths[field].strip():
            raise ValueError(f"Task config paths must define non-empty '{field}': {label}")

    objects = config.get("objects")
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError(f"Task config must define a non-empty 'objects' mapping: {label}")
    for object_name, metadata in objects.items():
        if not isinstance(object_name, str) or not isinstance(metadata, Mapping):
            raise ValueError(f"Task config object entries must be mappings: {label}")
        for field in _REQUIRED_OBJECT_FIELDS:
            if field not in metadata:
                raise ValueError(
                    f"Task config object {object_name!r} is missing '{field}': {label}"
                )
        for field in ("original_id", "category", "usd_path"):
            if not isinstance(metadata[field], str) or not metadata[field].strip():
                raise ValueError(
                    f"Task config object {object_name!r} has invalid '{field}': {label}"
                )
        for field in ("usd_scale", "position"):
            if not _is_xyz(metadata[field]):
                raise ValueError(
                    f"Task config object {object_name!r} must define numeric XYZ '{field}': {label}"
                )

    episodes = config.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Task config must define a non-empty 'episodes' list: {label}")
    for index, episode in enumerate(episodes):
        episode_label = f"{label} episode {index}"
        if not isinstance(episode, Mapping):
            raise ValueError(f"Task config episode must be a mapping: {episode_label}")
        for field in ("task_id", "task_description", "src", "dest", "target_object_id"):
            if not isinstance(episode.get(field), str) or not episode[field].strip():
                raise ValueError(
                    f"Task config episode must define non-empty '{field}': {episode_label}"
                )

        placements = episode.get("placements")
        if not isinstance(placements, Mapping) or not placements:
            raise ValueError(
                f"Task config episode must define a non-empty 'placements' mapping: {episode_label}"
            )
        unknown_objects = set(placements) - set(objects)
        if unknown_objects:
            raise ValueError(
                f"Task config episode references unknown objects {sorted(unknown_objects)}: "
                f"{episode_label}"
            )

        target_at_source = False
        for object_name, placement in placements.items():
            if not isinstance(placement, Mapping):
                raise ValueError(
                    f"Task placement {object_name!r} must be a mapping: {episode_label}"
                )
            original_id = placement.get("original_id")
            furniture = placement.get("furniture")
            if not isinstance(original_id, str) or not isinstance(furniture, str):
                raise ValueError(
                    f"Task placement {object_name!r} must define original_id and furniture: "
                    f"{episode_label}"
                )
            if objects[object_name]["original_id"] != original_id:
                raise ValueError(
                    f"Task placement {object_name!r} disagrees with object metadata: {episode_label}"
                )
            if original_id == episode["target_object_id"] and furniture == episode["src"]:
                target_at_source = True
        if not target_at_source:
            raise ValueError(f"Task target object is not placed at its source: {episode_label}")

    return config


def resolve_repo_path(value: str | Path) -> Path:
    """Expand environment variables and resolve a repo-relative path."""
    raw_value = str(value)
    referenced_variables = {
        match.group("braced") or match.group("plain")
        for match in _ENV_VAR_PATTERN.finditer(raw_value)
    }
    missing_variables = sorted(
        variable for variable in referenced_variables if not os.environ.get(variable)
    )
    if missing_variables:
        raise ValueError(
            "Path references unset environment variable(s): " + ", ".join(missing_variables)
        )

    path = Path(os.path.expandvars(raw_value)).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def load_task_config(path: str | Path | None = None) -> dict:
    """Load a task config without requiring machine-specific files."""
    config_path = resolve_repo_path(path or os.environ.get("DEMO_TASK_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=_YAML_LOADER) or {}

    return validate_task_config(config, config_path)


def get_metadata_path(filename: str) -> Path:
    """Path to a bundled metadata file (in repo metadata/ dir)."""
    return _PROJECT_ROOT / "metadata" / filename
