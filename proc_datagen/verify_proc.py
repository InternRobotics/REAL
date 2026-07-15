"""
verify_proc.py — 对 task_generator.py 输出的 per-scene per-type YAML 文件
进行物理验证。

环境变量:
  TASK_SOURCE_PATH  — 输入 YAML 文件路径（如 proc_datagen/configs/{scene_id}/{task_type}.yaml）
  OUTPUT_PATH       — 输出目录（会自动创建）

用法:
  # 普通模式（验证该文件中全部任务）
  TASK_SOURCE_PATH=proc_datagen/configs/MVUCSQAKTKJ5EAABAAAAABQ8/interactive.yaml \
  OUTPUT_PATH=proc_datagen/verify_results/interactive/MVUCSQAKTKJ5EAABAAAAABQ8 \
  python proc_datagen/verify_proc.py

  # 调试模式（只跑前 N 个任务）
  ... python proc_datagen/verify_proc.py --max-tasks 20
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

import yaml


# ── YAML formatting helpers (shared with task_generator.py) ────────────────


class _LiteralStr(str):
    pass


class _FlowList(list):
    pass


def _represent_literal(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _represent_flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _make_yaml_dumper():
    class _CustomDumper(yaml.Dumper):
        pass

    _CustomDumper.add_representer(_LiteralStr, _represent_literal)
    _CustomDumper.add_representer(_FlowList, _represent_flow_list)
    return _CustomDumper


# ======================================================================
# 环境变量 / 路径配置
# ======================================================================
TASK_SOURCE_PATH = os.environ["TASK_SOURCE_PATH"]
OUTPUT_PATH = Path(os.environ["OUTPUT_PATH"])
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# 共用日志工具
# ======================================================================
class TeeLogger:
    """同时写入 stdout 和文件的日志工具"""

    def __init__(self, filepath):
        self._file = open(filepath, "a", buffering=1)
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stdout, name)


# ======================================================================
# 加载 YAML 并解析任务
# ======================================================================
def load_yaml_doc(path: str) -> dict:
    """Load and return the YAML document."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_tasks(yaml_doc: dict, max_tasks: int | None = None) -> list:
    """
    从 YAML doc 中解析 episodes，为每条 episode 重建
    computed_placements 和 obj_meta，并附上 original_idx / task_id。
    """
    objects = yaml_doc.get("objects", {})
    episodes = yaml_doc.get("episodes", [])

    tasks = []
    for i, episode in enumerate(episodes):
        placements_raw = episode.get("placements", {})
        if not placements_raw:
            print(f"  [WARN] episode #{i} has no placements — skipping")
            continue

        # 重建 computed_placements
        computed_placements = {}
        for obj_key, p_info in placements_raw.items():
            obj_data = objects.get(obj_key)
            if obj_data is None:
                print(
                    f"  [WARN] episode #{i}: obj_key '{obj_key}' not in objects — skipping placement"
                )
                continue
            original_id = p_info["original_id"]
            furniture = p_info["furniture"]
            key = f"{original_id}_on_{furniture}"
            computed_placements[key] = {
                "position": list(obj_data["position"]),
                "furniture": furniture,
                "original_id": original_id,
            }

        if not computed_placements:
            print(f"  [WARN] episode #{i} has no valid placements — skipping")
            continue

        task = {
            "original_idx": i,
            "task_id": episode.get("task_id", f"proc_task_{i}"),
            "computed_placements": computed_placements,
            "episode": episode,  # keep original episode for output
        }
        tasks.append(task)

    print(f"[load_tasks] {len(tasks)} tasks from YAML (out of {len(episodes)} episodes total)")

    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]
        print(f"[load_tasks] Capped to {len(tasks)} tasks (--max-tasks {max_tasks})")

    return tasks


# ======================================================================
# 资产 / 场景库加载
# ======================================================================
def load_libraries(scene_id: str):
    """Load asset and furniture libraries for the given scene."""
    asset_lib_path = str(
        _PROJECT_ROOT / "assets/metadata/consolidated_asset_library_with_size.json"
    )
    furniture_lib_path = str(
        _PROJECT_ROOT / "assets/metadata" / scene_id / "scene_furniture_library.json"
    )

    print(f"[init] Loading asset libraries...", flush=True)
    asset_lib = json.load(open(asset_lib_path))
    scene_anno = json.load(open(furniture_lib_path))

    # Resolve relative usd_path entries using MESATASK_USD_ROOT env var.
    _mesatask_root = os.environ.get("MESATASK_USD_ROOT", "")
    if _mesatask_root:
        for _entry in asset_lib.values():
            _p = _entry.get("usd_path", "")
            if _p and not os.path.isabs(_p):
                _entry["usd_path"] = os.path.join(_mesatask_root, _p)

    # original_uid → library key 映射
    original_id2valid_prim = {info["original_uid"]: vid for vid, info in asset_lib.items()}

    return asset_lib, scene_anno, original_id2valid_prim


# ======================================================================
# 仿真初始化
# ======================================================================
def init_simulation(scene_id: str, scene_anno: dict):
    """初始化 internutopia 仿真环境（无机器人，只加载场景家具）"""
    from internutopia_extension.configs.objects import InteractiveObjCfg
    from internutopia_extension.configs.tasks import FiniteStepTaskCfg
    from internutopia.core.config import Config, SimConfig
    from internutopia.core.gym_env import Env
    from internutopia_extension import import_extensions

    scene_usd = str(_PROJECT_ROOT / f"assets/scenes/{scene_id}_usd/scene.usd")
    assert os.path.exists(scene_usd), f"Scene USD not found: {scene_usd}"

    furnitures = [
        InteractiveObjCfg(**{k: info[k] for k in ("name", "prim_path", "components")})
        for uid, info in scene_anno.items()
    ]

    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            headless=True,
            webrtc=False,
        ),
        task_configs=[
            FiniteStepTaskCfg(
                scene_asset_path=scene_usd,
                scene_scale=(0.01, 0.01, 0.01),
                robots=[],
                objects=furnitures,
                max_steps=int(1e6),
            ),
        ],
    )

    env = Env(config)
    import_extensions()
    env.reset()
    return env


# ======================================================================
# 物理验证
# ======================================================================
_INVALID_USD_PRIM_COMPONENT = re.compile(r"[^A-Za-z0-9_]")


def _allocate_usd_prim_component(raw_name: str, used_names: set[str]) -> str:
    """Return a unique, ASCII-safe USD prim path component.

    USD prim components cannot start with a digit and cannot contain path
    separators, whitespace, or punctuation.  ``used_names`` belongs to one
    verification run so names that collapse to the same sanitized value get a
    stable numeric suffix instead of referring to the same simulator object.
    """
    component = _INVALID_USD_PRIM_COMPONENT.sub("_", str(raw_name))
    if not component:
        component = "obj"
    if component[0].isdigit():
        component = f"obj_{component}"

    candidate = component
    suffix = 2
    while candidate in used_names:
        candidate = f"{component}_{suffix}"
        suffix += 1

    used_names.add(candidate)
    return candidate


def physics_verify(
    env,
    tasks: list,
    asset_lib: dict,
    original_id2valid_prim: dict,
    settle_steps: int = 500,
    fall_threshold: float = 0.3,
) -> tuple[list, list]:
    """
    对每个任务：按 computed_placements 中的坐标 spawn 物体，
    运行若干步仿真，检查物体是否掉落或飞出。
    返回: (passed_tasks, failed_tasks)
    """
    from internutopia_extension.configs.objects import InteractiveObjCfg
    from internutopia.core.scene.object import create_object

    passed_tasks = []
    failed_tasks = []
    current_objects = {}  # spawn_name → info dict，用于跨 task 清理
    used_spawn_names = set()  # Simulator objects remain registered across tasks.

    for task in tqdm(tasks, desc="Physics verification"):
        task_id = task["task_id"]
        original_idx = task["original_idx"]
        placements = task["computed_placements"]

        # ---- 清理上一轮的物体（移到场景外） ----
        for spawn_name in list(current_objects.keys()):
            try:
                obj = env.runner.current_tasks[env._current_task_name].objects.get(spawn_name)
                if obj is not None:
                    obj.prim.set_world_pose(position=(-100, -100, 0))
            except Exception:
                pass
        current_objects = {}

        # ---- Spawn 物体 ----
        initial_positions = {}
        spawn_error = None

        for obj_name, info in placements.items():
            # Keep the task/YAML object key untouched; only the internal USD prim
            # component needs an identifier-safe, run-unique name.
            raw_spawn_name = f"{obj_name}_t{original_idx}"
            spawn_name = _allocate_usd_prim_component(raw_spawn_name, used_spawn_names)
            x, y, z = info["position"]
            original_id = info["original_id"]

            # 尝试将 original_id 映射到 valid_prim_id
            if original_id in original_id2valid_prim:
                valid_id = original_id2valid_prim[original_id]
            elif original_id in asset_lib:
                valid_id = original_id
            else:
                spawn_error = f"Object '{original_id}' not found in asset library"
                break

            obj_meta = asset_lib[valid_id]
            usd_path = obj_meta["usd_path"]
            scale = obj_meta["usd_scale"]
            category = max(obj_meta["all_categories"], key=lambda k: obj_meta["all_categories"][k])

            obj_cfg = InteractiveObjCfg(
                name=spawn_name,
                prim_path=f"/World/env_0/scene/Meshes/{spawn_name}",
                usd_path=str(usd_path),
                # 放在计算位置略高处，等物理稳定后落下
                position=(x, y, z + 0.1),
                # X +90°：Y-up → Z-up  [w, x, y, z]
                orientation=(0.7071068, 0.7071068, 0.0, 0.0),
                collider=True,
                components={"graspable": f"/World/env_0/scene/Meshes/{spawn_name}"},
                scale=scale,
            )

            try:
                _obj = create_object(obj_cfg)
                sim_task = env.runner.current_tasks[env._current_task_name]
                _obj.set_up_scene(sim_task._scene)
                sim_task.objects[spawn_name] = _obj
                current_objects[spawn_name] = {
                    "original_key": obj_name,  # 用于还原输出 key
                    "category": category,
                    "original_id": original_id,
                    "initial_pos": (x, y, z + 0.1),
                }
                initial_positions[obj_name] = (x, y, z + 0.1)
            except Exception as e:
                spawn_error = f"Failed to create object '{obj_name}' (spawn_name={spawn_name}): {e}"
                break

        if spawn_error:
            print(f"  [FAIL][spawn] {task_id} (orig={original_idx}): {spawn_error}")
            failed_tasks.append(
                {
                    "original_idx": original_idx,
                    "task_id": task_id,
                    "error": spawn_error,
                    "stage": "physics_spawn",
                }
            )
            continue

        # ---- Warm-up：让 PhysX 注册新碰撞体 ----
        for _ in range(30):
            env.step([{}])

        # ---- 稳定仿真 ----
        for _ in range(settle_steps):
            env.step([{}])

        # ---- 检查最终位置 ----
        failed_objects = []
        final_positions = {}

        for spawn_name, info in current_objects.items():
            out_key = info["original_key"]  # 还原为原始 obj_name，保持输出格式不变
            try:
                obj = env.runner.current_tasks[env._current_task_name].objects.get(spawn_name)
                if obj is None:
                    failed_objects.append((out_key, "object not found after simulation"))
                    continue

                final_pos, _ = obj.get_world_pose()
                initial_z = info["initial_pos"][2]
                final_z = float(final_pos[2])

                final_positions[out_key] = {
                    "initial_pos": list(info["initial_pos"]),
                    "final_pos": [float(final_pos[0]), float(final_pos[1]), final_z],
                    "category": info["category"],
                    "original_id": info["original_id"],
                }

                if final_z < initial_z - fall_threshold:
                    failed_objects.append((out_key, f"fell: z {initial_z:.3f} → {final_z:.3f}"))
                elif abs(float(final_pos[0])) > 50 or abs(float(final_pos[1])) > 50 or final_z < -1:
                    failed_objects.append((out_key, f"out of bounds: {list(final_pos)}"))

            except Exception as e:
                failed_objects.append((out_key, f"check error: {e}"))

        if failed_objects:
            err = "; ".join(f"{out_key}: {r}" for out_key, r in failed_objects)
            print(f"  [FAIL][physics] {task_id} (orig={original_idx}): {err}")
            failed_tasks.append(
                {
                    "original_idx": original_idx,
                    "task_id": task_id,
                    "error": err,
                    "stage": "physics_settle",
                    "placements": placements,
                    "final_positions": final_positions,
                }
            )
        else:
            print(
                f"  [PASS] {task_id} (orig={original_idx})  "
                f"objects={[info['original_key'] for info in current_objects.values()]}"
            )
            passed_tasks.append(
                {
                    "original_idx": original_idx,
                    "task_id": task_id,
                    "placements": placements,
                    "final_positions": final_positions,
                }
            )

    print(f"\nPhysics verification — Passed: {len(passed_tasks)}  Failed: {len(failed_tasks)}")
    return passed_tasks, failed_tasks


# ======================================================================
# YAML 输出
# ======================================================================
def save_results_yaml(
    yaml_doc: dict,
    passed_tasks: list,
    failed_tasks: list,
    passed_file: Path,
    failed_file: Path,
):
    """
    将验证结果保存为 YAML 文件。
    passed YAML 保持与输入相同的结构（scene_id, paths, objects, episodes），
    只保留通过的 episodes，并在每个 episode 中添加 final_positions。
    """
    Dumper = _make_yaml_dumper()

    # Build set of passed episode indices and their final_positions
    passed_idx_map = {}  # original_idx -> final_positions
    for pt in passed_tasks:
        passed_idx_map[pt["original_idx"]] = pt.get("final_positions", {})

    # Filter episodes
    passed_episodes = []
    for i, episode in enumerate(yaml_doc.get("episodes", [])):
        if i in passed_idx_map:
            ep = dict(episode)
            ep["final_positions"] = passed_idx_map[i]
            passed_episodes.append(ep)

    # Collect objects referenced by passed episodes
    all_objects = yaml_doc.get("objects", {})
    referenced_obj_keys = set()
    for ep in passed_episodes:
        for obj_key in ep.get("placements", {}):
            referenced_obj_keys.add(obj_key)
    passed_objects = {k: v for k, v in all_objects.items() if k in referenced_obj_keys}

    passed_doc = {
        "scene_id": yaml_doc["scene_id"],
        "task_type": yaml_doc.get("task_type", ""),
        "paths": yaml_doc.get("paths", {}),
        "objects": passed_objects,
        "episodes": passed_episodes,
    }

    with open(passed_file, "w") as f:
        yaml.dump(
            passed_doc,
            f,
            Dumper=Dumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    # Failed: simple list of error records
    failed_doc = {
        "scene_id": yaml_doc["scene_id"],
        "task_type": yaml_doc.get("task_type", ""),
        "failed_episodes": failed_tasks,
    }

    with open(failed_file, "w") as f:
        yaml.dump(
            failed_doc,
            f,
            Dumper=Dumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics verification for proc_taskgen tasks (YAML input/output)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Only verify the first N tasks (useful for debugging)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=500,
        help="Number of physics steps to settle objects (default: 500)",
    )
    parser.add_argument(
        "--fall-threshold",
        type=float,
        default=0.3,
        help="Z-drop threshold to classify an object as 'fallen' (default: 0.3 m)",
    )
    args = parser.parse_args()

    # 输出文件
    passed_file = OUTPUT_PATH / "physics_passed.yaml"
    failed_file = OUTPUT_PATH / "physics_failed.yaml"
    log_file = OUTPUT_PATH / "physics_log.txt"

    # 重定向 stdout 同时写入日志文件
    tee = TeeLogger(log_file)
    sys.stdout = tee

    # 1. 加载 YAML
    print("=" * 60)
    print(f"verify_proc.py — Physics Verification (YAML)")
    print(f"  Task file    : {TASK_SOURCE_PATH}")
    print(f"  Output       : {OUTPUT_PATH}")
    print(f"  Max tasks    : {args.max_tasks or 'all'}")
    print(f"  Settle steps : {args.settle_steps}")
    print(f"  Fall threshold: {args.fall_threshold} m")
    print("=" * 60)

    yaml_doc = load_yaml_doc(TASK_SOURCE_PATH)
    scene_id = yaml_doc["scene_id"]
    print(f"  Scene        : {scene_id}")
    print(f"  Task type    : {yaml_doc.get('task_type', 'unknown')}")

    tasks = load_tasks(yaml_doc, max_tasks=args.max_tasks)
    if not tasks:
        print("No tasks to verify — exiting.")
        sys.exit(0)

    # 2. 加载资产库
    asset_lib, scene_anno, original_id2valid_prim = load_libraries(scene_id)

    # 3. 初始化仿真
    print("\nInitializing simulation...")
    env = init_simulation(scene_id, scene_anno)

    # 4. 物理验证
    print("\nRunning physics verification...\n")
    passed, failed = physics_verify(
        env,
        tasks,
        asset_lib,
        original_id2valid_prim,
        settle_steps=args.settle_steps,
        fall_threshold=args.fall_threshold,
    )

    # 5. 保存结果
    save_results_yaml(yaml_doc, passed, failed, passed_file, failed_file)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"  physics_passed.yaml : {len(passed)}")
    print(f"  physics_failed.yaml : {len(failed)}")
    print("\nDone!")

    tee.close()
