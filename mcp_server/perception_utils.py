"""
Perception utilities for visual prompting and object detection.
Uses replicator for rendering instead of obs_dict.
"""

import os
from PIL import Image as PILImage
import numpy as np
from collections import defaultdict
from typing import Dict
from dataclasses import dataclass

import omni.replicator.core as rep
from openai import OpenAI

_embedding_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_CACHE: dict = {}
_embedding_model_checked = False


def _mask_api_key_prefix() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "<EMPTY>"
    return api_key[:8]


def _ensure_embedding_model_available() -> None:
    global _embedding_model_checked
    if _embedding_model_checked:
        return
    try:
        _embedding_client.embeddings.create(
            input="health_check",
            model=EMBEDDING_MODEL,
        )
        _embedding_model_checked = True
    except Exception as exc:
        key_prefix = _mask_api_key_prefix()
        print(f"[embedding_check] unavailable, api_key_prefix={key_prefix}")
        raise RuntimeError(
            f"Embedding model '{EMBEDDING_MODEL}' is unavailable. api_key_prefix={key_prefix}"
        ) from exc

_ensure_embedding_model_available()

@dataclass
class VisualPromptingComponent:
    """Component for visual prompting."""
    object_name: str
    component_name: str
    component_type: str
    prim_path: str


# Global annotators - will be initialized on first use
_rgb_annotator = None
_instance_seg_annotator = None
_render_product = None


def init_annotators(camera, resolution=(640, 480)):
    """Initialize replicator annotators for RGB and instance segmentation."""
    global _rgb_annotator, _instance_seg_annotator, _render_product

    if _render_product is None:
        _render_product = rep.create.render_product(camera, resolution)

        _rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        _rgb_annotator.attach([_render_product])

        _instance_seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
        _instance_seg_annotator.attach([_render_product])


def get_rgb_and_segmentation():
    """Get RGB image and instance segmentation from replicator."""
    rep.orchestrator.step(rt_subframes=4)

    rgb_data = _rgb_annotator.get_data()
    instance_seg_data = _instance_seg_annotator.get_data()

    return rgb_data, instance_seg_data


def _find_objects_exact(
    target_category: str,
    current_assets: dict,
    camera=None,
    predefined_markers: dict = None,
) -> tuple:
    """
    Find and highlight objects matching the target category (exact match).

    Args:
        target_category: The category to search for (GT category, exact match)
        current_assets: Dict mapping object names to their metadata (category, original_id)
        camera: Camera object for rendering (only needed for first call to init)
        predefined_markers: Optional dict mapping object_name -> marker_id.
                           If provided, use these marker IDs instead of auto-generating.
                           This ensures consistency with walkaround text markers.

    Returns:
        (result_image, marker2component) - PIL Image with markers and dict mapping marker IDs to components
    """
    from internutopia_extension.utils.som import draw_mask_and_number_on_image

    # Initialize annotators if needed
    if _render_product is None and camera is not None:
        init_annotators(camera)

    # Get current render
    rgb_data, instance_seg = get_rgb_and_segmentation()

    # Convert to proper formats
    image_rgb = rgb_data[..., :3].astype(np.uint8)
    observation = PILImage.fromarray(image_rgb)

    mask_data = instance_seg['data']
    id_to_labels = instance_seg['info']['idToLabels']
    prim_to_id = {v: int(k) for k, v in id_to_labels.items()}

    # Find target objects by exact category match
    target_objs = []
    for asset_name, asset_info in current_assets.items():
        if asset_info['category'] == target_category:
            target_objs.append(asset_name)

    # Find instance IDs for target objects
    grounded_prims = defaultdict(list)
    for scene_prim_path, instance_id in prim_to_id.items():
        for target_obj in target_objs:
            target_prefix = f"/World/env_0/scene/Meshes/{target_obj}"
            if scene_prim_path.startswith(target_prefix):
                grounded_prims[target_obj].append(instance_id)
                break

    masks = []
    labels = []
    marker2component: Dict[str, VisualPromptingComponent] = {}

    if predefined_markers is not None:
        # Use predefined marker IDs for consistency with walkaround
        for obj_name, marker_id in predefined_markers.items():
            if obj_name not in grounded_prims:
                continue  # Object not visible in current view

            instance_ids = grounded_prims[obj_name]
            if not instance_ids:
                continue

            instance_mask = np.isin(mask_data, instance_ids)
            if not instance_mask.any():
                continue

            masks.append(instance_mask)
            labels.append(str(marker_id))
            marker2component[str(marker_id)] = VisualPromptingComponent(
                object_name=obj_name,
                component_name="NONE",
                component_type='mesh',
                prim_path=obj_name
            )
    else:
        # Auto-generate marker IDs (sorted by instance_id)
        sorted_grounded_prims = sorted(
            grounded_prims.items(), key=lambda item: next(iter(item[1]), 0)
        )

        label_index = 1
        for obj_name, instance_ids in sorted_grounded_prims:
            if not instance_ids:
                continue

            instance_mask = np.isin(mask_data, instance_ids)
            if not instance_mask.any():
                continue

            masks.append(instance_mask)
            labels.append(str(label_index))
            marker2component[str(label_index)] = VisualPromptingComponent(
                object_name=obj_name,
                component_name="NONE",
                component_type='mesh',
                prim_path=obj_name
            )
            label_index += 1

    if len(masks) > 0:
        result_image_array = draw_mask_and_number_on_image(
            image_rgb, masks, labels, anno_mode=["Mask", "Mark"], alpha=0.25
        )
        result_image = PILImage.fromarray(result_image_array)
        return result_image, marker2component
    else:
        return observation, {}


def _get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text, using cache."""
    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]
    embd = _embedding_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    vec = np.array(embd.data[0].embedding)
    EMBEDDING_CACHE[text] = vec
    return vec


def find_objects(
    target_category: str,
    current_assets: dict,
    camera=None,
    predefined_markers: dict = None,
) -> tuple:
    """
    Find and highlight objects matching the target category with visual markers,
    using embedding similarity for fuzzy category matching.

    When the agent's query (target_category) does not exactly match any
    category in current_assets, this function computes cosine similarity
    between the query embedding and all asset category embeddings, then
    selects the closest match.

    Args:
        target_category: The category to search for (may be approximate)
        current_assets: Dict mapping object names to their metadata (category, original_id)
        camera: Camera object for rendering (only needed for first call to init)
        predefined_markers: Optional dict mapping object_name -> marker_id.

    Returns:
        (result_image, marker2component) - PIL Image with markers and dict mapping marker IDs to components
    """
    from internutopia_extension.utils.som import draw_mask_and_number_on_image

    # Initialize annotators if needed
    if _render_product is None and camera is not None:
        init_annotators(camera)

    # Get current render
    rgb_data, instance_seg = get_rgb_and_segmentation()

    # Convert to proper formats
    image_rgb = rgb_data[..., :3].astype(np.uint8)
    observation = PILImage.fromarray(image_rgb)

    mask_data = instance_seg['data']
    id_to_labels = instance_seg['info']['idToLabels']
    prim_to_id = {v: int(k) for k, v in id_to_labels.items()}

    # Collect all unique categories from current assets and cache their embeddings
    current_categories = set()
    for asset_info in current_assets.values():
        cat = asset_info['category']
        current_categories.add(cat)
        _get_embedding(cat)

    # Get embedding for target category and find best match
    target_embd_vec = _get_embedding(target_category)

    similarities = {}
    for category in current_categories:
        cat_vec = EMBEDDING_CACHE[category]
        similarity = np.dot(target_embd_vec, cat_vec) / (
            np.linalg.norm(target_embd_vec) * np.linalg.norm(cat_vec)
        )
        similarities[category] = similarity

    matched_category = max(similarities, key=similarities.get)
    print(f"[find_objects] query='{target_category}' -> matched='{matched_category}' "
          f"(sim={similarities[matched_category]:.4f})")

    # Find target objects by matched category
    target_objs = []
    for asset_name, asset_info in current_assets.items():
        if asset_info['category'] == matched_category:
            target_objs.append(asset_name)

    # Find instance IDs for target objects
    grounded_prims = defaultdict(list)
    for scene_prim_path, instance_id in prim_to_id.items():
        for target_obj in target_objs:
            target_prefix = f"/World/env_0/scene/Meshes/{target_obj}"
            if scene_prim_path.startswith(target_prefix):
                grounded_prims[target_obj].append(instance_id)
                break

    masks = []
    labels = []
    marker2component: Dict[str, VisualPromptingComponent] = {}

    if predefined_markers is not None:
        for obj_name, marker_id in predefined_markers.items():
            if obj_name not in grounded_prims:
                continue
            instance_ids = grounded_prims[obj_name]
            if not instance_ids:
                continue
            instance_mask = np.isin(mask_data, instance_ids)
            if not instance_mask.any():
                continue
            masks.append(instance_mask)
            labels.append(str(marker_id))
            marker2component[str(marker_id)] = VisualPromptingComponent(
                object_name=obj_name,
                component_name="NONE",
                component_type='mesh',
                prim_path=obj_name
            )
    else:
        sorted_grounded_prims = sorted(
            grounded_prims.items(), key=lambda item: next(iter(item[1]), 0)
        )
        label_index = 1
        for obj_name, instance_ids in sorted_grounded_prims:
            if not instance_ids:
                continue
            instance_mask = np.isin(mask_data, instance_ids)
            if not instance_mask.any():
                continue
            masks.append(instance_mask)
            labels.append(str(label_index))
            marker2component[str(label_index)] = VisualPromptingComponent(
                object_name=obj_name,
                component_name="NONE",
                component_type='mesh',
                prim_path=obj_name
            )
            label_index += 1

    if len(masks) > 0:
        result_image_array = draw_mask_and_number_on_image(
            image_rgb, masks, labels, anno_mode=["Mask", "Mark"], alpha=0.25
        )
        result_image = PILImage.fromarray(result_image_array)
        return result_image, marker2component
    else:
        return observation, {}


def highlight_receptacles(
    furniture_prims: list,
    furniture_names: list,
    hidden_target: str = '',
    camera=None,
) -> tuple:
    """
    Show receptacles/furniture with visual markers.

    Args:
        furniture_prims: List of furniture prim paths
        furniture_names: List of furniture names
        hidden_target: Name of target receptacle (for warning if not visible)
        camera: Camera object for rendering (only needed for first call to init)

    Returns:
        (result_image, marker2component) - PIL Image with markers and dict mapping marker IDs to components
    """
    from internutopia_extension.utils.som import draw_mask_and_number_on_image

    # Initialize annotators if needed
    if _render_product is None and camera is not None:
        init_annotators(camera)

    # Build prim to furniture name mapping
    prim2fur = {}
    prim2fur['/World/env_0/objects/sink_0_base'] = 'sink_0'
    prim2fur['/World/env_0/objects/sink_1_base'] = 'sink_1'
    prim2fur['/World/env_0/objects/washingmachine_1_base'] = 'washingmachine_1'
    prim2fur['/World/env_0/scene/Meshes/Animation/electriccooker'] = 'electriccooker_0'
    for i in range(len(furniture_names)):
        prim2fur[furniture_prims[i]] = furniture_names[i]

    # Get current render
    rgb_data, instance_seg = get_rgb_and_segmentation()

    # Convert to proper formats
    image_rgb = rgb_data[..., :3].astype(np.uint8)
    observation = PILImage.fromarray(image_rgb)

    mask_data = instance_seg['data']
    id_to_labels = instance_seg['info']['idToLabels']
    prim_to_id = {v: int(k) for k, v in id_to_labels.items()}

    # Find instance IDs for furniture
    grounded_prims = defaultdict(list)
    for scene_prim_path, instance_id in prim_to_id.items():
        for prim, fur_name in prim2fur.items():
            if scene_prim_path.startswith(prim):
                grounded_prims[fur_name].append(instance_id)
                break

    sorted_grounded_prims = sorted(
        grounded_prims.items(), key=lambda item: next(iter(item[1]), 0)
    )

    masks = []
    labels = []
    label_index = 1
    marker2component: Dict[str, VisualPromptingComponent] = {}

    for fur_name, instance_ids in sorted_grounded_prims:
        if not instance_ids:
            continue

        instance_mask = np.isin(mask_data, instance_ids)

        if not instance_mask.any():
            if fur_name == hidden_target:
                print(f"Warning: receptacle {fur_name} has no visible instance in the current view.")
            continue

        masks.append(instance_mask)
        labels.append(str(label_index))
        marker2component[str(label_index)] = VisualPromptingComponent(
            object_name=fur_name,
            component_name="NONE",
            component_type='mesh',
            prim_path=fur_name
        )
        label_index += 1

    if len(masks) > 0:
        result_image_array = draw_mask_and_number_on_image(
            image_rgb, masks, labels, anno_mode=["Mask", "Mark"], alpha=0.5
        )
        result_image = PILImage.fromarray(result_image_array)
        return result_image, marker2component
    else:
        return observation, {}


def render_persisted_markers(
    predefined_markers: Dict[str, int],
) -> tuple:
    """
    Re-render persisted marker overlays on the current view.

    Unlike find_objects which filters by category first,
    this function renders markers for specific objects by name,
    regardless of category. Used for marker map persistence after focus_on.

    Args:
        predefined_markers: Dict mapping object_name -> marker_id

    Returns:
        (result_image, marker2component) - PIL Image with markers and
        dict mapping marker IDs to VisualPromptingComponent
    """
    from internutopia_extension.utils.som import draw_mask_and_number_on_image

    rgb_data, instance_seg = get_rgb_and_segmentation()

    image_rgb = rgb_data[..., :3].astype(np.uint8)
    observation = PILImage.fromarray(image_rgb)

    mask_data = instance_seg['data']
    id_to_labels = instance_seg['info']['idToLabels']
    prim_to_id = {v: int(k) for k, v in id_to_labels.items()}

    # Find instance IDs for each named object
    grounded_prims = defaultdict(list)
    for scene_prim_path, instance_id in prim_to_id.items():
        for obj_name in predefined_markers:
            target_prefix = f"/World/env_0/scene/Meshes/{obj_name}"
            if scene_prim_path.startswith(target_prefix):
                grounded_prims[obj_name].append(instance_id)
                break

    masks = []
    labels = []
    marker2component: Dict[str, VisualPromptingComponent] = {}

    for obj_name, marker_id in predefined_markers.items():
        if obj_name not in grounded_prims:
            continue
        instance_ids = grounded_prims[obj_name]
        if not instance_ids:
            continue
        instance_mask = np.isin(mask_data, instance_ids)
        if not instance_mask.any():
            continue
        masks.append(instance_mask)
        labels.append(str(marker_id))
        marker2component[str(marker_id)] = VisualPromptingComponent(
            object_name=obj_name,
            component_name="NONE",
            component_type='mesh',
            prim_path=obj_name,
        )

    if len(masks) > 0:
        result_image_array = draw_mask_and_number_on_image(
            image_rgb, masks, labels, anno_mode=["Mask", "Mark"], alpha=0.25
        )
        result_image = PILImage.fromarray(result_image_array)
        return result_image, marker2component
    else:
        return observation, {}


def get_rgb_image() -> np.ndarray:
    """Get current RGB image from replicator."""
    rep.orchestrator.step(rt_subframes=4)
    rgb_data = _rgb_annotator.get_data()
    return rgb_data[..., :3]
