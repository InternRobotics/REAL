"""
Environment setup for evaluation — non-headless GUI mode.

All configuration is driven by a single YAML task config file.
No hardcoded paths — every path either comes from the config file or
can be overridden with an environment variable.

Usage:
    DEMO_TASK_CONFIG=configs/demo_task.yaml \\
    python -m mcp_server.mcp_server_demo

Config file (DEMO_TASK_CONFIG, default: configs/demo_task.yaml):
    See configs/demo_task.yaml for the full schema.

Optional env-var overrides:
    TARGET_SCENE_ID   — override scene_id from config
    TRAJ_PATH         — output directory (default: eval_output_demo)
    SCENE_USD_PATH    — override paths.scene_usd in config
    EMPTY_USD_PATH    — path to an empty scene USD; required when USE_EMPTY_SCENE=1
    USE_EMPTY_SCENE   — set to 1 to use EMPTY_USD_PATH instead of the scene USD
    LIFT_USD_PATH     — robot USD; required when USE_LIFT_ROBOT=1
    USE_LIFT_ROBOT    — set to 1 to spawn the lift robot
    SCENE_ANNO_PATH   — path to scene caption JSON (optional, skipped if absent)
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import yaml

# =============================================================================
# Load task config
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

_cfg_path = Path(os.environ.get(
    'DEMO_TASK_CONFIG',
    str(_REPO_ROOT / 'configs' / 'demo_task.yaml'),
)).expanduser()
if not _cfg_path.is_absolute():
    _cfg_path = _REPO_ROOT / _cfg_path
assert _cfg_path.exists(), f"Task config not found: {_cfg_path}"

with open(_cfg_path) as _f:
    _task_cfg = yaml.safe_load(_f)


def _resolve(p: str) -> Path:
    """Return path as-is if absolute, otherwise relative to repo root."""
    p = Path(p)
    return p if p.is_absolute() else _REPO_ROOT / p


# =============================================================================
# Configuration
# =============================================================================

TARGET_SCENE_ID = os.environ.get('TARGET_SCENE_ID', _task_cfg['scene_id'])
TRAJ_PATH = _resolve(os.environ.get('TRAJ_PATH', 'eval_output_demo'))

_paths = _task_cfg['paths']

# Scene USD
USE_EMPTY_SCENE = os.environ.get('USE_EMPTY_SCENE', '0') == '1'
if USE_EMPTY_SCENE:
    _empty_usd = os.environ.get('EMPTY_USD_PATH', '')
    assert _empty_usd, "USE_EMPTY_SCENE=1 requires the EMPTY_USD_PATH env var"
    SCENE_USD_PATH = _empty_usd
else:
    SCENE_USD_PATH = str(
        os.environ.get('SCENE_USD_PATH', '') or _resolve(_paths['scene_usd'])
    )

assert os.path.exists(SCENE_USD_PATH), f"Scene USD not found: {SCENE_USD_PATH}"

# Occupancy map directory
OCC_MAP_PATH: Path = _resolve(_paths['occ_map_dir'])
assert OCC_MAP_PATH.exists(), f"Occ map dir not found: {OCC_MAP_PATH}"

print(f"[mcp_env_debug] config          = {_cfg_path}")
print(f"[mcp_env_debug] SCENE_USD_PATH  = {SCENE_USD_PATH}")
print(f"[mcp_env_debug] OCC_MAP_PATH    = {OCC_MAP_PATH}")

# =============================================================================
# Object registry — built from objects section in config
# =============================================================================

_object_registry: dict = {}
for _obj_name, _obj_cfg in _task_cfg.get('objects', {}).items():
    _entry = dict(_obj_cfg)
    _entry['usd_path'] = str(_resolve(_obj_cfg['usd_path']))
    _object_registry[_obj_name] = _entry

# =============================================================================
# Furniture and scene setup
# =============================================================================

from internutopia_extension.configs.objects import InteractiveObjCfg, UsdObjCfg

_furniture_lib_path = _resolve(_paths['furniture_lib'])
scene_anno = json.load(open(_furniture_lib_path))

# Scene captions are optional
_anno_path = os.environ.get('SCENE_ANNO_PATH', '')
try:
    all_captions = json.load(open(_anno_path)) if _anno_path else {}
    if not _anno_path:
        print("[mcp_env_debug] SCENE_ANNO_PATH not set, skipping captions.")
except FileNotFoundError:
    print(f"[mcp_env_debug] WARNING: captions not found at {_anno_path}, skipping.")
    all_captions = {}

furnitures = []
furniture_names = []
furniture_prims = []
object_per_room = defaultdict(list)

empty_world_graph = {}
for furniture_uid, furniture_info in scene_anno.items():
    if TARGET_SCENE_ID != furniture_info['scene_id']:
        continue
    used_keys = ["name", "prim_path", "components"]
    furniture_data = {k: furniture_info[k] for k in used_keys}
    furniture = InteractiveObjCfg(**furniture_data)
    assert '/Root' not in str(furniture_data)
    furnitures.append(furniture)
    furniture_names.append(furniture_info['name'])
    empty_world_graph[furniture_info['name']] = {'content': []}
    if 'door' in furniture_info['components']:
        empty_world_graph[furniture_info['name']]['door'] = False
    furniture_prims.append(furniture_info['prim_path'])
    object_per_room[furniture_info['room_name']].append(furniture_info['name'])

# =============================================================================
# Lift robot definition (optional, enabled by USE_LIFT_ROBOT=1)
# =============================================================================

from typing import Optional as _Optional
from internutopia.core.config import RobotCfg
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.robot.articulation import IArticulation
from internutopia.core.scene.scene import IScene


class LiftCfg(RobotCfg):
    name: _Optional[str] = 'lift'
    type: _Optional[str] = 'Lift'
    prim_path: _Optional[str] = '/lift'
    usd_path: _Optional[str] = None  # must be provided via LIFT_USD_PATH


@BaseRobot.register('Lift')
class LiftRobot(BaseRobot):
    def __init__(self, config: LiftCfg, scene: IScene):
        super().__init__(config, scene)
        self.articulation = IArticulation.create(
            prim_path=config.prim_path,
            name=config.name,
            usd_path=config.usd_path,
            position=np.array(config.position),
        )

    def post_reset(self):
        super().post_reset()
        print("[LiftRobot] joints:", self.articulation.dof_names)
        from pxr import UsdPhysics, Usd, PhysxSchema
        from omni.isaac.core.utils.stage import get_current_stage
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(self.config.prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Set(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

    def apply_action(self, action):
        if not isinstance(action, dict):
            return
        for controller_name, controller_action in action.items():
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.articulation.apply_action(control)

    def get_obs(self):
        position, orientation = self.articulation.get_pose()
        controllers_obs, sensors_obs = super()._get_controllers_and_sensors_obs()
        obs = {'position': position, 'orientation': orientation,
               'controllers': controllers_obs, 'sensors': sensors_obs}
        obs["joint_velocitis"] = self.articulation.get_joint_velocities()
        obs["joint_positions"] = self.articulation.get_joint_positions()
        return self._make_ordered(obs)


USE_LIFT_ROBOT = os.environ.get('USE_LIFT_ROBOT', '0') == '1'

if USE_LIFT_ROBOT:
    _lift_usd = os.environ.get('LIFT_USD_PATH', '')
    assert _lift_usd, "USE_LIFT_ROBOT=1 requires the LIFT_USD_PATH env var"
    lift_cfg = LiftCfg(
        usd_path=_lift_usd,
        position=(1.6, -1.3, 0.0),
        controllers=[],
        sensors=[],
    )

# =============================================================================
# Simulation setup — NON-HEADLESS (GUI window)
# =============================================================================

from internutopia_extension.configs.tasks import FiniteStepTaskCfg
from internutopia.core.config import Config, SimConfig
from internutopia.core.gym_env import Env
from internutopia_extension import import_extensions

config = Config(
    simulator=SimConfig(physics_dt=1 / 240, rendering_dt=1 / 240, use_fabric=False, headless=False, webrtc=False),
    task_configs=[
        FiniteStepTaskCfg(
            scene_asset_path=SCENE_USD_PATH,
            scene_scale=(0.01, 0.01, 0.01),
            robots=[lift_cfg] if USE_LIFT_ROBOT else [],
            objects=furnitures,
            max_steps=int(1e10),
        ),
    ],
)

env = Env(config)
import_extensions()
obs, _ = env.reset()
print(f'========INIT OBS{obs}=============')

from internutopia.core.scene.object import create_object

import omni.replicator.core as rep

camera = rep.create.camera(
    name="Camera_0",
    position=(1.4, -0.7, 1.5),
    clipping_range=(0.01, 1000),
)

from internutopia.core.util.omni_usd_util import compute_path_bbox
from internutopia_extension.utils.occ_map import NavManager
from internutopia_extension.utils.camera_utils import (
    track_object,
    calculate_look_at_quaternion_fixed_camera,
)
from omni.isaac.core.prims.xform_prim import XFormPrim

camera_prim = XFormPrim(prim_path="/Replicator/Camera_0_Xform")

# Patch NavManager to load occupancy.npy from OCC_MAP_PATH
_nav_occ_file = OCC_MAP_PATH / "occupancy.npy"
_orig_nav_init = NavManager.__init__
def _patched_nav_init(self, scene_id: str):
    import numpy as _np
    self.scene_id = scene_id
    self.occupancy_map = _np.load(str(_nav_occ_file))
    self.x_corrds = self.occupancy_map[0, 1:]
    self.y_corrds = self.occupancy_map[1:, 0]
    self.free_map = self.occupancy_map[1:, 1:]
    self.x_res = _np.mean(_np.diff(self.x_corrds))
    self.y_res = _np.mean(_np.diff(self.y_corrds))
    if self.x_res <= 0 or self.y_res <= 0:
        self.x_res = abs(self.x_res)
        self.y_res = abs(self.y_res)
    self.x_origin, self.y_origin = self.x_corrds[0], self.y_corrds[0]
    self.H, self.W = self.free_map.shape
NavManager.__init__ = _patched_nav_init

nav_manager = NavManager(scene_id=TARGET_SCENE_ID)

# =============================================================================
# Build processed_eval_episodes from config episodes
# =============================================================================

def _build_world_graph(placements: dict) -> dict:
    world_graph = deepcopy(empty_world_graph)
    for obj_name, placement_info in placements.items():
        furniture_name = placement_info['furniture']
        if furniture_name in world_graph:
            world_graph[furniture_name]['content'].append(obj_name)
    return world_graph


def _find_target_object_name(placements: dict, target_obj_id: str, src: str) -> str:
    fallback = None
    for obj_name, info in placements.items():
        if info['original_id'] != target_obj_id:
            continue
        if fallback is None:
            fallback = obj_name
        if info.get('furniture') == src:
            return obj_name
    return fallback


def query_object_category(obj_id: str) -> str:
    for obj in _object_registry.values():
        if obj['original_id'] == obj_id:
            return obj['category']
    raise KeyError(f"Object id {obj_id} not found in object registry")


processed_eval_episodes = []
for _ep_idx, _ep in enumerate(_task_cfg.get('episodes', [])):
    _placements = _ep['placements']
    _target_id = _ep['target_object_id']
    _src = _ep['src']

    _target_name = _find_target_object_name(_placements, _target_id, _src)
    _target_category = query_object_category(_target_id)

    processed_eval_episodes.append({
        'task_id': _ep['task_id'],
        'original_idx': _ep_idx,
        'task_description': _ep['task_description'],
        'initial_world_graph': _build_world_graph(_placements),
        'placements': _placements,
        'execution_plan': _ep.get('execution_plan', []),
        'src': _src,
        'dest': _ep['dest'],
        'src_distractors': _ep.get('src_distractors', []),
        'dest_distractors': _ep.get('dest_distractors', []),
        'obj_distractors': _ep.get('obj_distractors', []),
        'obj_distractor_meta': _ep.get('obj_distractor_meta', {}),
        'target_object_id': _target_id,
        'target_object_name': _target_name,
        'target_category': _target_category,
    })

print(f"[mcp_env_debug] Loaded eval episodes: {len(processed_eval_episodes)}")


# =============================================================================
# Object spawning — world-space positions from config
# =============================================================================

def spawn_objects_by_world_graph(env: Env, episode: dict, current_objects: dict):
    """
    Spawn objects at world-space positions defined in the config's objects section.
    A +0.4 m Z offset is applied so objects drop onto the surface from above.
    """
    placements = episode['placements']

    # Clean up existing objects
    for current_obj_name in current_objects.keys():
        current_obj = env.runner.current_tasks[
            env._current_task_name
        ].objects.get(current_obj_name)
        if current_obj is not None:
            current_obj.prim.set_world_pose(position=(-100, -100, 0))

    current_objects = {}

    task = env.runner.current_tasks[env._current_task_name]
    for obj_name, placement_info in placements.items():
        obj_meta = _object_registry[obj_name]
        pos = obj_meta['position']
        spawn_pos = (pos[0], pos[1], pos[2] + 0.4)

        print(f"[spawn] {obj_name} ({obj_meta['category']}) -> {spawn_pos}")

        obj_cfg = InteractiveObjCfg(
            name=obj_name,
            prim_path=f"/World/env_0/scene/Meshes/{obj_name}",
            usd_path=obj_meta['usd_path'],
            position=spawn_pos,
            orientation=(0.7071068, 0.7071068, 0.0, 0.0),
            collider=True,
            components={"graspable": f"/World/env_0/scene/Meshes/{obj_name}"},
            scale=obj_meta['usd_scale'],
        )

        _obj = create_object(obj_cfg)
        _obj.set_up_scene(task._scene)
        task.objects[obj_name] = _obj
        current_objects[obj_name] = {
            "category": obj_meta['category'],
            "original_id": placement_info['original_id'],
        }

    for _ in range(50):
        env.step(action=[{}])

    return current_objects
