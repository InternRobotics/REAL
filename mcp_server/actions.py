"""
Action handlers for evaluation.
Each handler is a plain function that takes EvalState + env and returns
(result_type, result_data) where result_type is "text" or "image".

Follows the patterns from replay/mcp4sft_sync_refactored.py.
"""

import base64
import json
import os
from io import BytesIO
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image as PILImage
from pxr import UsdPhysics

from internutopia.core.util.omni_usd_util import compute_path_bbox
from internutopia_extension.utils.camera_utils import (
    track_object,
    calculate_look_at_quaternion_fixed_camera,
)
from mcp_server.mcp_env import (
    nav_manager,
    camera_prim,
    object_per_room,
    furniture_names as ALL_FURNITURES,
    furniture_prims as FURNITURE_PRIMS,
)
from mcp_server.perception_utils import (
    find_objects,
    highlight_receptacles,
    render_persisted_markers,
    get_rgb_image,
)


# =============================================================================
# State
# =============================================================================

@dataclass
class EvalState:
    """Mutable evaluation state, replacing global variables."""
    current_obs_dict: Any = None
    current_marker_map: Optional[Dict[str, str]] = None
    current_landmark: Optional[str] = None
    current_inv: Optional[str] = None
    current_pos: Optional[tuple] = None
    current_extra_assets: Dict = field(default_factory=dict)
    world_graph: Dict = field(default_factory=dict)
    camera_orientation: Any = None  # Last known camera orientation
    persist_marker_map_on_gaze: bool = False
    precomputed_nav_positions: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)




# =============================================================================
# Helpers
# =============================================================================

def _ensure_playing():
    """Resume the Isaac Sim timeline if it was stopped (e.g. after USD stage edits)."""
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()


def step_simulation(env, n_steps: int = 1):
    """Step the simulation and return the last observation."""
    _ensure_playing()
    obs = None
    for _ in range(n_steps):
        obs, *_ = env.step([{}])
    return obs


def _set_rigid_body_enabled(prim_path: str, enabled: bool):
    """Enable or disable rigid body physics via USD API (avoids tensor view issues)."""
    from omni.isaac.core.utils.stage import get_current_stage
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(enabled)


def compute_nav_target_params(prim_path: str) -> Tuple[tuple, float, float]:
    """Compute navigation target parameters (center, radius, lower_bound).

    Returns None for all values if the bbox is degenerate (e.g. empty prim).
    """
    obj_min_point, obj_max_point = compute_path_bbox(prim_path)
    min_x, min_y, min_z = obj_min_point
    max_x, max_y, max_z = obj_max_point

    # Detect degenerate bbox (e.g. FLT_MAX values from empty/invalid prims)
    if abs(min_x) > 1e30 or abs(max_x) > 1e30 or abs(min_y) > 1e30 or abs(max_y) > 1e30:
        return None, None, None

    target_center = (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2,
    )
    target_radius = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5 / 2
    radius_lower_bound = min(max_x - min_x, max_y - min_y) / 2

    return target_center, target_radius, radius_lower_bound


def compute_placement_position(
    obj_bbox: tuple,
    surface_bbox: tuple,
    robot_pos: tuple,
    margin_factor: float = 4.0,
    z_offset: float = 0.15,
) -> List[float]:
    """
    Compute optimal placement position using clamp algorithm.
    Places object as close to robot position as possible within surface bounds.
    """
    obj_min, obj_max = obj_bbox
    surf_min, surf_max = surface_bbox

    obj_half_x = (obj_max[0] - obj_min[0]) / 2.0
    obj_half_y = (obj_max[1] - obj_min[1]) / 2.0
    obj_half_z = (obj_max[2] - obj_min[2]) / 2.0

    valid_min_x = surf_min[0] + margin_factor * obj_half_x
    valid_max_x = surf_max[0] - margin_factor * obj_half_x
    valid_min_y = surf_min[1] + margin_factor * obj_half_y
    valid_max_y = surf_max[1] - margin_factor * obj_half_y

    robot_x, robot_y, _ = robot_pos

    if valid_min_x > valid_max_x:
        target_x = (surf_min[0] + surf_max[0]) / 2.0
    else:
        target_x = max(valid_min_x, min(robot_x, valid_max_x))

    if valid_min_y > valid_max_y:
        target_y = (surf_min[1] + surf_max[1]) / 2.0
    else:
        target_y = max(valid_min_y, min(robot_y, valid_max_y))

    target_z = surf_max[2] + obj_half_z + z_offset

    return [target_x, target_y, target_z]


def get_rgb_observation() -> np.ndarray:
    """Get RGB image from replicator annotator."""
    return get_rgb_image()

def image_to_base64(pil_image: PILImage) -> str:
    """Convert a PIL image to base64-encoded PNG string."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def rgb_array_to_base64(rgb_array: np.ndarray) -> str:
    """Convert an RGB numpy array to base64-encoded PNG string."""
    pil_image = PILImage.fromarray(rgb_array.astype(np.uint8))
    return image_to_base64(pil_image)


def get_debug_info(state: EvalState) -> dict:
    """Serialize current state for debug output."""
    wg_summary = ''
    for fur in state.world_graph:
        content = state.world_graph[fur].get('content', [])
        if len(content) > 0:
            wg_summary += f"{fur}: {content}\n"

    return {
        'CURRENT_LANDMARK': state.current_landmark,
        'CURRENT_POS': list(state.current_pos) if state.current_pos is not None else None,
        'CURRENT_INV': state.current_inv,
        'CURRENT_MARKER_MAP': state.current_marker_map,
        # Canonical machine-readable world graph for downstream clients.
        'world_graph': deepcopy(state.world_graph),
        # Backward-compatible human-readable summary kept for debugging.
        'WORLD_GRAPH': wg_summary,
    }


# =============================================================================
# Action Handlers
# =============================================================================

def handle_list_receptacles(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """List all receptacles by room."""
    state.persist_marker_map_on_gaze = False
    receptacle_info = ""
    for room, furnitures in object_per_room.items():
        receptacle_info += f"Room: {room}\n"
        for furniture in furnitures:
            receptacle_info += f"  - {furniture}\n"
    return "text", receptacle_info


def handle_find_objects(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Detect and highlight objects of a category in the current view."""
    result_image, marker2component = find_objects(
        arguments['target_category'],
        state.current_extra_assets,
    )
    state.current_marker_map = {
        k: v.object_name for k, v in marker2component.items()
    }
    state.persist_marker_map_on_gaze = True
    return "image", image_to_base64(result_image)


def handle_highlight_receptacles(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Highlight all receptacle objects in the current view."""
    state.persist_marker_map_on_gaze = False
    result_image, marker2component = highlight_receptacles(
        furniture_prims=FURNITURE_PRIMS,
        furniture_names=ALL_FURNITURES,
    )
    state.current_marker_map = {
        k: v.object_name for k, v in marker2component.items()
    }
    return "image", image_to_base64(result_image)


def handle_navigate_to(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Navigate camera to a furniture receptacle."""
    state.persist_marker_map_on_gaze = False
    state.current_marker_map = None
    target_furniture_name = arguments['receptacle_name']

    # Eval/Test mode: use precomputed nav positions when available.
    # Action-level is_test has higher priority than global default.
    
    nav_target = env.runner.current_tasks[env._current_task_name].objects.get(target_furniture_name)
    if nav_target is None:
        return "text", f"Receptacle {target_furniture_name} not found in the scene."

    surface_component = nav_target.components.get("top_shelf")
    door_component = nav_target.components.get("door")

    if surface_component is None and door_component is None:
        return "text", f"Object {target_furniture_name} has no valid navigable component."

    # Find navigable component prim path
    prim_path = None
    for comp in [door_component, surface_component]:
        if comp is not None:
            prim_path = comp.prim_path
            if 'Constraint' in prim_path:
                prim_path = prim_path.replace('Constraint', 'Group')
            break

    target_center, target_radius, radius_lower_bound = compute_nav_target_params(prim_path)

    # Fallback: if component bbox is degenerate, use the parent object's prim
    if target_center is None:
        print(f"  Component bbox degenerate for {prim_path}, falling back to parent: {nav_target.prim_path}")
        prim_path = nav_target.prim_path
        target_center, target_radius, radius_lower_bound = compute_nav_target_params(prim_path)

    if target_center is None:
        return "text", f"Cannot compute bounding box for {target_furniture_name}."

    # --- Debug info ---
    print(f"  [nav_debug] furniture={target_furniture_name}")
    print(f"  [nav_debug] components={list(nav_target.components.keys())}")
    print(f"  [nav_debug] prim_path={prim_path}")
    print(f"  [nav_debug] parent_prim_path={nav_target.prim_path}")
    print(f"  [nav_debug] target_center={target_center}, target_radius={target_radius:.3f}, radius_lower_bound={radius_lower_bound:.3f}")
    obj_min_point, obj_max_point = compute_path_bbox(prim_path)
    print(f"  [nav_debug] bbox_min={tuple(obj_min_point)}, bbox_max={tuple(obj_max_point)}")
    # Also print parent bbox for comparison
    if prim_path != nav_target.prim_path:
        p_min, p_max = compute_path_bbox(nav_target.prim_path)
        print(f"  [nav_debug] parent_bbox_min={tuple(p_min)}, parent_bbox_max={tuple(p_max)}")
    # --- End debug ---

    # Use the furniture category path (common ancestor) for occlusion testing,
    # so that sibling objects of the same type (e.g. two washingmachines next to
    # each other) are not treated as occluders.
    parent_prim = nav_target.prim_path
    # Go one level up from e.g. ".../washingmachine/model_xxx" to ".../washingmachine/"
    category_prim_path = parent_prim.rsplit('/', 1)[0] + '/'

    nav_position = nav_manager.get_camera_position_by_seg(
        target_component_prim_path=category_prim_path,
        target_x=target_center[0],
        target_y=target_center[1],
        target_z=target_center[2],
        target_radius=target_radius,
        radius_lower_bound=radius_lower_bound,
        camera_prim=camera_prim,
        debug_file_name=f"nav_test_output/debug_{target_furniture_name}",
    )

    if nav_position is None:
        return "text", f"Cannot find a non-occluded position to navigate to {target_furniture_name}."

    camera_orientation = calculate_look_at_quaternion_fixed_camera(nav_position, target_center)

    camera_prim.set_world_pose(
        position=(nav_position[0], nav_position[1], nav_position[2]),
        orientation=camera_orientation,
    )

    state.current_landmark = target_furniture_name
    state.current_pos = nav_position
    state.camera_orientation = camera_orientation

    step_simulation(env, 50)

    return "image", rgb_array_to_base64(get_rgb_observation())


def handle_explore_receptacle(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """List objects on current furniture."""
    if state.current_landmark is None:
        return "text", "No landmark selected. Use 'navigate_to' first."

    content = state.world_graph.get(state.current_landmark, {}).get('content', [])
    if len(content) == 0:
        return "text", f"Seems there is no object on the {state.current_landmark}'s top shelf to observe."

    walkaround_result = {}
    nl_result = "I found the following object(s): \n"
    label_index = 1
    for obj_name in content:
        if obj_name not in state.current_extra_assets:
            continue
        obj_category = state.current_extra_assets[obj_name]['category']
        walkaround_result[str(label_index)] = obj_name
        nl_result += f"\t{label_index}: a(an) {obj_category}.\n"
        label_index += 1

    state.current_marker_map = walkaround_result
    state.persist_marker_map_on_gaze = True

    return "text", nl_result


def handle_focus_on(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Focus camera on a specific marker/object."""
    if state.current_marker_map is None:
        return "text", "No marker map available. Use 'find_objects' or 'explore_receptacle' first."

    marker_id = str(arguments['marker_id'])
    if marker_id not in state.current_marker_map:
        return "text", f"Marker ID {marker_id} not found in the marker map."

    obj_name = state.current_marker_map[marker_id]
    gaze_target = env.runner.current_tasks[
        env._current_task_name
    ].objects.get(obj_name)

    if gaze_target is None:
        return "text", f"Object {obj_name} is not available. Cannot focus on it."

    comp = gaze_target.components.get("graspable")
    if comp is None:
        return "text", f"Object {obj_name} has no graspable component. Cannot focus on it."

    target_center, target_radius, radius_lower_bound = compute_nav_target_params(comp.prim_path)

    nav_position = nav_manager.get_camera_position_snug(
        target_component_prim_path=comp.prim_path,
        target_x=target_center[0],
        target_y=target_center[1],
        target_radius=target_radius,
        radius_lower_bound=0,
        debug_file_name=None,
    )

    if nav_position is None:
        nav_position = state.current_pos

    if nav_position is None:
        return "text", "No camera position available. Use 'navigate_to' first."

    camera_orientation = state.camera_orientation
    if camera_orientation is None:
        return "text", "No camera orientation available. Use 'navigate_to' first."

    camera_prim.set_world_pose(
        position=(nav_position[0], nav_position[1], nav_position[2]),
        orientation=camera_orientation,
    )

    track_object(
        camera_prim=camera_prim,
        target_object=gaze_target.prim_path,
        camera_position=nav_position,
    )

    state.current_pos = nav_position

    step_simulation(env, 50)

    if state.persist_marker_map_on_gaze and state.current_marker_map is not None:
        predefined_markers = {
            obj_name: int(marker_id)
            for marker_id, obj_name in state.current_marker_map.items()
        }
        result_image, marker2component = render_persisted_markers(
            predefined_markers=predefined_markers,
        )
        state.current_marker_map = {
            k: v.object_name for k, v in marker2component.items()
        }
        return "image", image_to_base64(result_image)

    return "image", rgb_array_to_base64(get_rgb_observation())


def handle_pick(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Pick up an object by marker id."""
    state.persist_marker_map_on_gaze = False
    if state.current_inv is not None:
        return "text", f"You are already holding '{state.current_inv}'. Place it before picking another object."
    if state.current_marker_map is None:
        return "text", "No marker map available. Use 'find_objects', 'explore_receptacle' or 'highlight_receptacles' first."

    marker_id = str(arguments['marker_id'])
    if marker_id not in state.current_marker_map:
        return "text", f"Marker ID {marker_id} not found in the marker map."

    target_name = state.current_marker_map[marker_id]
    obj = env.runner.current_tasks[
        env._current_task_name
    ].objects.get(target_name)

    if obj is None:
        return "text", f"Object {target_name} is not graspable. Cannot pick it up."

    target_component = obj.components.get("graspable")
    if target_component is None:
        return "text", f"Object {target_name} has no graspable component."

    # Execute pick
    _set_rigid_body_enabled(target_component.prim_path, False)
    obj.set_world_pose((-100, -100, 0))
    step_simulation(env, 20)
    obj.set_visibility(visible=False)

    # Update world graph
    for furniture_name in state.world_graph:
        if target_name in state.world_graph[furniture_name].get('content', []):
            state.world_graph[furniture_name]['content'] = list(
                set(state.world_graph[furniture_name]['content']) - {target_name}
            )
            break

    state.current_inv = target_name
    print(f"[DEBUG] Picked up object: {target_name}")

    step_simulation(env, 500)

    return "image", rgb_array_to_base64(get_rgb_observation())


def handle_place(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Place held object on a receptacle surface."""
    state.persist_marker_map_on_gaze = False
    if state.current_marker_map is None:
        return "text", "No marker map available. Use 'highlight_receptacles' first."

    marker_id = str(arguments['marker_id'])
    if marker_id not in state.current_marker_map:
        return "text", f"Marker ID {marker_id} not found in the marker map."

    if state.current_inv is None:
        return "text", "No object in the inventory to place."

    target_name = state.current_marker_map[marker_id]

    if (
        target_name not in state.world_graph
        or 'potlid' in target_name.lower()
        or 'content' not in state.world_graph[target_name]
    ):
        return "text", f"'{target_name}' is not a valid placement surface."

    if state.world_graph[target_name].get('door') is False:
        return "text", f"The door of '{target_name}' is closed. Use 'open' first."

    target_surface_furniture = env.runner.current_tasks[env._current_task_name].objects.get(target_name)
    if target_surface_furniture is None:
        return "text", f"Target furniture {target_name} not found in the scene."

    surface_comp = target_surface_furniture.components.get("top_shelf")
    if surface_comp is None:
        return "text", f"Furniture {target_name} has no top_shelf component."

    target_object = env.runner.current_tasks[
        env._current_task_name
    ].objects.get(state.current_inv)
    if target_object is None:
        return "text", f"Held object '{state.current_inv}' not found in the scene."

    obj_bbox = compute_path_bbox(target_object.components.get("graspable").prim_path)
    surf_bbox = compute_path_bbox(surface_comp.prim_path)

    placement_pos = compute_placement_position(obj_bbox, surf_bbox, state.current_pos)

    # Execute placement
    _, rot = target_object.get_world_pose()
    _set_rigid_body_enabled(target_object.components.get("graspable").prim_path, True)
    target_object.set_world_pose(placement_pos, rot)
    target_object.set_visibility(visible=True)
    target_object.components.get("graspable").attached = False

    # Update world graph
    state.world_graph[target_name]['content'].append(state.current_inv)
    state.current_inv = None

    step_simulation(env, 1000)

    return "image", rgb_array_to_base64(get_rgb_observation())


def handle_open(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Open a door of an articulated object."""
    state.persist_marker_map_on_gaze = False
    if state.current_marker_map is None:
        return "text", "No marker map available."

    marker_id = str(arguments['marker_id'])
    if marker_id not in state.current_marker_map:
        return "text", f"Marker ID {marker_id} not found in the marker map."

    target_name = state.current_marker_map[marker_id]

    try:
        if (
            'door' in state.world_graph.get(target_name, {})
            and state.world_graph[target_name]['door'] is False
        ):
            target_door_furniture = env.runner.current_tasks[env._current_task_name].objects.get(target_name)
            door_comp = target_door_furniture.components.get("door")
            door_comp.set_angle(90)
            state.world_graph[target_name]['door'] = True
    except Exception as e:
        print(f"[WARN] Error opening door: {e}")

    step_simulation(env, 100)
    return "image", rgb_array_to_base64(get_rgb_observation())


def handle_close(
    state: EvalState, env, arguments: dict
) -> Tuple[str, str]:
    """Close a door of an articulated object."""
    state.persist_marker_map_on_gaze = False
    if state.current_marker_map is None:
        return "text", "No marker map available."

    marker_id = str(arguments['marker_id'])
    if marker_id not in state.current_marker_map:
        return "text", f"Marker ID {marker_id} not found in the marker map."

    target_name = state.current_marker_map[marker_id]

    try:
        if (
            'door' in state.world_graph.get(target_name, {})
            and state.world_graph[target_name]['door'] is True
        ):
            target_door_furniture = env.runner.current_tasks[env._current_task_name].objects.get(target_name)
            door_comp = target_door_furniture.components.get("door")
            door_comp.set_angle(0)
            state.world_graph[target_name]['door'] = False
    except Exception as e:
        print(f"[WARN] Error closing door: {e}")

    step_simulation(env, 100)

    return "image", rgb_array_to_base64(get_rgb_observation())


# =============================================================================
# Dispatch
# =============================================================================

ACTION_HANDLERS = {
    "list_receptacles": handle_list_receptacles,
    "find_objects": handle_find_objects,
    "highlight_receptacles": handle_highlight_receptacles,
    "navigate_to": handle_navigate_to,
    "explore_receptacle": handle_explore_receptacle,
    "focus_on": handle_focus_on,
    "pick": handle_pick,
    "place": handle_place,
    "open": handle_open,
    "close": handle_close,
}


def dispatch_action(
    action_name: str,
    arguments: dict,
    state: EvalState,
    env,
) -> Tuple[str, str, dict]:
    """
    Dispatch an action to the appropriate handler.

    Returns:
        (result_type, result_data, debug_info)
        result_type: "text" or "image"
        result_data: text string or base64-encoded PNG
        debug_info: dict with current state for debugging
    """
    handler = ACTION_HANDLERS.get(action_name)
    if handler is None:
        result_type, result_data = "text", f"Action '{action_name}' not recognized."
    else:
        result_type, result_data = handler(state, env, arguments)

    debug_info = get_debug_info(state)
    return result_type, result_data, debug_info
