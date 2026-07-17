"""Load and validate the released REAL-Bench task definitions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mcp_server.config import load_task_config

DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmark"
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

_REQUIRED_TASK_FIELDS: dict[str, type] = {
    "benchmark_task_id": str,
    "family": str,
    "global_index": int,
    "scene_id": str,
    "benchmark_instruction": str,
    "task_id": str,
    "initial_world_graph": dict,
    "goal_world_graph": dict,
    "placements": dict,
    "execution_plan": list,
}


class RealBenchValidationError(ValueError):
    """Raised when a REAL-Bench bundle violates its published contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RealBenchValidationError(message)


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        raise RealBenchValidationError(f"Required benchmark file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_YAML_LOADER)
    except yaml.YAMLError as error:
        raise RealBenchValidationError(f"Invalid YAML in {path}: {error}") from error


def _validate_object_metadata(task: dict[str, Any], task_id: str) -> set[str]:
    referenced_usds: set[str] = set()
    for field in ("obj_meta", "obj_distractor_meta"):
        metadata = task.get(field, {})
        _require(isinstance(metadata, dict), f"{task_id}: {field} must be an object")
        for object_id, entry in metadata.items():
            _require(
                isinstance(entry, dict), f"{task_id}: metadata for {object_id} must be an object"
            )
            usd_path = entry.get("usd_path")
            if usd_path is None:
                continue
            _require(isinstance(usd_path, str), f"{task_id}: usd_path must be a string")
            _require(
                Path(usd_path).name == usd_path and usd_path.endswith(".usd"),
                f"{task_id}: usd_path must be a portable USD basename, got {usd_path!r}",
            )
            referenced_usds.add(usd_path)
    return referenced_usds


def _validate_object_views(task: dict[str, Any], task_id: str) -> None:
    placements = task["placements"]
    placement_locations = Counter(
        (placement["furniture"], placement["original_id"]) for placement in placements.values()
    )

    initial_locations: Counter[tuple[str, str]] = Counter()
    for furniture, state in task["initial_world_graph"].items():
        _require(
            isinstance(furniture, str) and isinstance(state, dict),
            f"{task_id}: initial world graph entries must be objects",
        )
        content = state.get("content", [])
        _require(
            isinstance(content, list) and all(isinstance(item, str) for item in content),
            f"{task_id}: initial world graph content must be a list of object IDs",
        )
        initial_locations.update((furniture, object_id) for object_id in content)
    _require(
        initial_locations == placement_locations,
        f"{task_id}: placements do not match the initial world graph",
    )

    for furniture, state in task["goal_world_graph"].items():
        _require(
            isinstance(furniture, str) and isinstance(state, dict),
            f"{task_id}: goal world graph entries must be objects",
        )
        content = state.get("content", [])
        _require(
            isinstance(content, list) and all(isinstance(item, str) for item in content),
            f"{task_id}: goal world graph content must be a list of object IDs",
        )

    metadata_ids = set(task.get("obj_meta", {})) | set(task.get("obj_distractor_meta", {}))
    placement_ids = {placement["original_id"] for placement in placements.values()}
    _require(
        placement_ids <= metadata_ids,
        f"{task_id}: placement object IDs are missing from object metadata",
    )

    final_positions = task.get("final_positions")
    if final_positions is not None:
        _require(isinstance(final_positions, dict), f"{task_id}: final_positions must be an object")
        _require(
            set(final_positions) == set(placements),
            f"{task_id}: final_positions keys do not match placements",
        )
        for placement_key, final_position in final_positions.items():
            _require(
                isinstance(final_position, dict)
                and final_position.get("original_id") == placements[placement_key]["original_id"],
                f"{task_id}: final position {placement_key!r} has the wrong object ID",
            )

    target_ids = list(task.get("obj_meta", {}))
    _require(len(target_ids) == 1, f"{task_id}: obj_meta must identify exactly one target")
    target_id = target_ids[0]
    _require(target_id in placement_ids, f"{task_id}: target object is not placed")

    source = task.get("src")
    destination = task.get("dest")
    _require(
        isinstance(source, str)
        and source in task["initial_world_graph"]
        and source in task["goal_world_graph"],
        f"{task_id}: source furniture is missing from a world graph",
    )
    _require(
        isinstance(destination, str)
        and destination in task["initial_world_graph"]
        and destination in task["goal_world_graph"],
        f"{task_id}: destination furniture is missing from a world graph",
    )
    _require(
        target_id in task["initial_world_graph"][source].get("content", []),
        f"{task_id}: target object is not initially at the source",
    )
    _require(
        target_id not in task["goal_world_graph"][source].get("content", []),
        f"{task_id}: target object remains at the source in the goal",
    )
    _require(
        target_id in task["goal_world_graph"][destination].get("content", []),
        f"{task_id}: target object is absent from the goal destination",
    )


def _validate_task(task: Any, index: int, allowed_families: set[str]) -> set[str]:
    _require(isinstance(task, dict), f"Task at index {index} must be an object")
    task_id = task.get("benchmark_task_id", f"index {index}")

    for field, expected_type in _REQUIRED_TASK_FIELDS.items():
        value = task.get(field)
        _require(
            isinstance(value, expected_type),
            f"{task_id}: {field} must be {expected_type.__name__}",
        )

    family = task["family"]
    _require(family in allowed_families, f"{task_id}: unknown family {family!r}")
    _require(task["global_index"] == index, f"{task_id}: expected global_index {index}")
    _require(
        task["benchmark_task_id"] == f"{family}/{task['task_id']}",
        f"{task_id}: benchmark_task_id must be '<family>/<task_id>'",
    )
    _require(task["scene_id"].strip() != "", f"{task_id}: scene_id must not be empty")
    _require(
        task["benchmark_instruction"].strip() != "",
        f"{task_id}: benchmark_instruction must not be empty",
    )

    placements = task["placements"]
    _require(placements, f"{task_id}: placements must not be empty")
    for placement_id, placement in placements.items():
        _require(
            isinstance(placement, dict),
            f"{task_id}: placement {placement_id!r} must be an object",
        )
        position = placement.get("position")
        _require(
            isinstance(position, list)
            and len(position) == 3
            and all(isinstance(coordinate, (int, float)) for coordinate in position),
            f"{task_id}: placement {placement_id!r} must have a numeric XYZ position",
        )
        for field in ("furniture", "original_id"):
            _require(
                isinstance(placement.get(field), str) and placement[field],
                f"{task_id}: placement {placement_id!r} must define {field}",
            )

    execution_plan = task["execution_plan"]
    _require(execution_plan, f"{task_id}: execution_plan must not be empty")
    for step_index, step in enumerate(execution_plan):
        _require(
            isinstance(step, dict)
            and isinstance(step.get("action"), str)
            and isinstance(step.get("args"), dict),
            f"{task_id}: execution_plan step {step_index} must define action and args",
        )

    referenced_usds = _validate_object_metadata(task, task_id)
    _validate_object_views(task, task_id)
    return referenced_usds


def _validate_runtime_view(config: dict[str, Any], task: dict[str, Any], path: Path) -> None:
    task_id = task.get("benchmark_task_id", path.stem)
    scene_id = task.get("scene_id")
    _require(config["scene_id"] == scene_id, f"{task_id}: runtime scene_id is inconsistent")

    expected_paths = {
        "scene_usd": f"assets/scenes/{scene_id}_usd/scene.usd",
        "occ_map_dir": f"assets/metadata/{scene_id}",
        "furniture_lib": f"assets/metadata/{scene_id}/scene_furniture_library.json",
    }
    for field, expected in expected_paths.items():
        _require(
            config["paths"][field] == expected,
            f"{task_id}: runtime path {field!r} is inconsistent",
        )

    placements = task.get("placements", {})
    objects = config["objects"]
    _require(
        set(objects) == set(placements),
        f"{task_id}: runtime objects do not match episode placements",
    )

    task_metadata = {**task.get("obj_meta", {}), **task.get("obj_distractor_meta", {})}
    for placement_name, placement in placements.items():
        object_config = objects[placement_name]
        original_id = placement.get("original_id")
        metadata = task_metadata.get(original_id)
        _require(
            isinstance(metadata, dict),
            f"{task_id}: runtime object {placement_name!r} has no episode metadata",
        )
        _require(
            object_config["original_id"] == original_id,
            f"{task_id}: runtime object {placement_name!r} has the wrong original_id",
        )
        _require(
            object_config["position"] == placement.get("position"),
            f"{task_id}: runtime object {placement_name!r} has the wrong position",
        )
        _require(
            object_config["category"] == metadata.get("category"),
            f"{task_id}: runtime object {placement_name!r} has the wrong category",
        )
        expected_usd = f"${{MESATASK_USD_ROOT}}/{metadata.get('usd_path')}"
        _require(
            object_config["usd_path"] == expected_usd,
            f"{task_id}: runtime object {placement_name!r} has the wrong USD path",
        )
        for field in ("size", "detailed_caption"):
            if field in metadata:
                _require(
                    object_config.get(field) == metadata[field],
                    f"{task_id}: runtime object {placement_name!r} has the wrong {field}",
                )

    target_ids = list(task.get("obj_meta", {}))
    _require(len(target_ids) == 1, f"{task_id}: obj_meta must identify exactly one target")
    _require(
        task.get("target_object_id") == target_ids[0],
        f"{task_id}: runtime target_object_id is inconsistent",
    )


def _load_episode_configs(
    root: Path,
    total_tasks: int,
    allowed_families: set[str],
) -> list[dict[str, Any]]:
    task_root = root / "tasks"
    _require(task_root.is_dir(), f"Required benchmark directory is missing: {task_root}")

    config_paths = sorted(task_root.rglob("*.yaml"))
    all_files = {path for path in task_root.rglob("*") if path.is_file()}
    _require(
        all_files == set(config_paths),
        "benchmark/tasks may contain only per-episode YAML files",
    )
    _require(
        len(config_paths) == total_tasks,
        f"Expected {total_tasks} per-episode YAML files, found {len(config_paths)}",
    )

    tasks: list[dict[str, Any]] = []
    for config_path in config_paths:
        _require(
            config_path.parent.parent == task_root,
            f"Episode config must be stored at tasks/<family>/<task_id>.yaml: {config_path}",
        )
        try:
            config = load_task_config(config_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise RealBenchValidationError(
                f"Invalid eval-server config {config_path}: {error}"
            ) from error

        episodes = config["episodes"]
        _require(len(episodes) == 1, f"{config_path}: expected exactly one episode")
        task = dict(episodes[0])
        family = task.get("family")
        task_name = task.get("task_id")
        _require(family in allowed_families, f"{config_path}: unknown family {family!r}")
        _require(
            config_path.parent.name == family and config_path.stem == task_name,
            f"{config_path}: path does not match episode family/task_id",
        )
        _require(
            isinstance(task.get("global_index"), int),
            f"{config_path}: global_index must be int",
        )
        _validate_runtime_view(config, task, config_path)
        tasks.append(task)

    tasks.sort(key=lambda task: task["global_index"])
    return tasks


def load_real_bench(
    root: str | Path = DEFAULT_BENCHMARK_ROOT,
    *,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """Load a REAL-Bench bundle after validating all 241 published tasks.

    Validation always covers the complete bundle. ``family`` only filters the
    returned list after every per-episode eval-server config has been checked.
    """

    benchmark_root = Path(root).expanduser().resolve()
    manifest = _read_yaml(benchmark_root / "manifest.yaml")

    _require(isinstance(manifest, dict), "manifest.yaml must contain a mapping")
    _require(manifest.get("benchmark") == "REAL-Bench", "Unexpected benchmark name")

    total_tasks = manifest.get("total_tasks")
    family_counts = manifest.get("family_counts")
    family_order = manifest.get("family_order")
    mesa_required_count = manifest.get("mesa_required_count")
    _require(isinstance(total_tasks, int) and total_tasks > 0, "Invalid total_tasks")
    _require(isinstance(family_counts, dict) and family_counts, "Invalid family_counts")
    _require(
        isinstance(family_order, list)
        and family_order
        and all(isinstance(name, str) for name in family_order),
        "Invalid family_order",
    )
    _require(set(family_counts) == set(family_order), "family_counts and family_order differ")
    _require(
        all(isinstance(count, int) and count > 0 for count in family_counts.values()),
        "Family counts must be positive integers",
    )
    _require(sum(family_counts.values()) == total_tasks, "Family counts do not sum to total_tasks")
    _require(
        isinstance(mesa_required_count, int) and mesa_required_count > 0,
        "Invalid mesa_required_count",
    )

    allowed_families = set(family_order)
    tasks = _load_episode_configs(benchmark_root, total_tasks, allowed_families)
    _require(len(tasks) == total_tasks, f"Expected {total_tasks} tasks, found {len(tasks)}")
    referenced_usds: set[str] = set()
    for index, task in enumerate(tasks):
        referenced_usds.update(_validate_task(task, index, allowed_families))

    mesa_required_path = benchmark_root / "mesa_required.txt"
    if not mesa_required_path.is_file():
        raise RealBenchValidationError(f"Required benchmark file is missing: {mesa_required_path}")
    mesa_required = [
        line.strip()
        for line in mesa_required_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(
        mesa_required == sorted(set(mesa_required)),
        "mesa_required.txt must contain sorted, unique USD basenames",
    )
    _require(
        len(mesa_required) == mesa_required_count,
        f"Expected {mesa_required_count} MesaTask basenames, found {len(mesa_required)}",
    )
    _require(
        set(mesa_required) == referenced_usds,
        "mesa_required.txt does not match task object metadata",
    )

    benchmark_ids = [task["benchmark_task_id"] for task in tasks]
    _require(
        len(set(benchmark_ids)) == len(benchmark_ids), "benchmark_task_id values must be unique"
    )
    actual_counts = Counter(task["family"] for task in tasks)
    _require(dict(actual_counts) == family_counts, "Task family counts do not match manifest")

    if family is None:
        return tasks

    normalized_family = family.upper()
    _require(
        normalized_family in allowed_families,
        f"Unknown family {family!r}; choose one of {', '.join(family_order)}",
    )
    return [task for task in tasks if task["family"] == normalized_family]
