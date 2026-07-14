#!/usr/bin/env python3
"""
All-in-One Pick-and-Place Task Generator.

Combines five task types in a single script sharing common infrastructure:
  - basic       : simple pick-and-place with furniture distractors
  - distractor  : same-category object distractors + detailed_caption grounding
  - articulation: tasks involving open/close of articulated furniture
  - interactive : same-purpose different-category distractors + fuzzy description
  - gather      : multi-source gather tasks (N objects to one destination)
"""

import json
import os
import re
import random
import itertools
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import asyncio as _asyncio
    from openai import AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ── Shared style templates ─────────────────────────────────────────────────────

STYLE_TEMPLATES = [
    "Pick up [obj] from [src] and place it on [dest].",
    "Transport [obj] at [src] to [dest].",
    "Bring [obj] from [src] to [dest].",
    "Move [obj] located at [src] to [dest].",
    "Retrieve [obj] from [src] and deposit it at [dest].",
    "Take [obj] from [src] and put it on [dest].",
    "Get [obj] from [src] and move it to [dest].",
    "Navigate to [src], grasp [obj], and carry it to [dest].",
    "Go to [src], pick up [obj], then travel to [dest] to place it.",
    "First, drive to [src] to find [obj], then deliver it to [dest].",
    "Head over to [src], fetch [obj], and bring it back to [dest].",
]

STYLE_TEMPLATES_DESC = [
    "Pick up [obj_desc] from [src] and place it on [dest].",
    "Transport [obj_desc] at [src] to [dest].",
    "Bring [obj_desc] from [src] to [dest].",
    "Move [obj_desc] located at [src] to [dest].",
    "Retrieve [obj_desc] from [src] and deposit it at [dest].",
    "Take [obj_desc] from [src] and put it on [dest].",
    "Get [obj_desc] from [src] and move it to [dest].",
    "Navigate to [src], grasp [obj_desc], and carry it to [dest].",
    "Go to [src], pick up [obj_desc], then travel to [dest] to place it.",
    "First, drive to [src] to find [obj_desc], then deliver it to [dest].",
    "Head over to [src], fetch [obj_desc], and bring it back to [dest].",
]

FUZZY_STYLE_TEMPLATES = [
    "Pick up a [purpose] from [src] and place it on [dest].",
    "Transport a [purpose] at [src] to [dest].",
    "Bring a [purpose] from [src] to [dest].",
    "Move a [purpose] located at [src] to [dest].",
    "Retrieve a [purpose] from [src] and deposit it at [dest].",
    "Take a [purpose] from [src] and put it on [dest].",
    "Get a [purpose] from [src] and move it to [dest].",
    "Navigate to [src], grasp a [purpose], and carry it to [dest].",
    "Go to [src], pick up a [purpose], then travel to [dest] to place it.",
    "First, drive to [src] to find a [purpose], then deliver it to [dest].",
    "Head over to [src], fetch a [purpose], and bring it back to [dest].",
]

GATHER_TEMPLATES_2 = [
    "Pick up [obj1_desc] from [src1] and [obj2_desc] from [src2], then place them both on [dest].",
    "Collect [obj1_desc] at [src1] and [obj2_desc] at [src2], and bring them to [dest].",
    "First grab [obj1_desc] from [src1], then get [obj2_desc] from [src2], and put them on [dest].",
    "Gather [obj1_desc] from [src1] and [obj2_desc] from [src2] onto [dest].",
    "Go to [src1] to get [obj1_desc], then to [src2] to get [obj2_desc], and deliver both to [dest].",
    "Retrieve [obj1_desc] from [src1] and [obj2_desc] from [src2], placing them on [dest].",
    "Take [obj1_desc] from [src1] and [obj2_desc] from [src2], and put them together on [dest].",
    "Head to [src1] for [obj1_desc] and [src2] for [obj2_desc], then bring them to [dest].",
]

GATHER_TEMPLATES_3 = [
    "Pick up [obj1_desc] from [src1], [obj2_desc] from [src2], and [obj3_desc] from [src3], then place them all on [dest].",
    "Collect [obj1_desc] at [src1], [obj2_desc] at [src2], and [obj3_desc] at [src3], and bring them to [dest].",
    "Gather [obj1_desc] from [src1], [obj2_desc] from [src2], and [obj3_desc] from [src3] onto [dest].",
    "Retrieve [obj1_desc] from [src1], [obj2_desc] from [src2], and [obj3_desc] from [src3], placing them all on [dest].",
    "First grab [obj1_desc] from [src1], then [obj2_desc] from [src2], then [obj3_desc] from [src3], and put them on [dest].",
]

GATHER_FUZZY_TEMPLATES_2 = [
    "Pick up a [purpose] from [src1] and another [purpose] from [src2], then place them both on [dest].",
    "Collect the [purpose]s from [src1] and [src2], and bring them to [dest].",
    "Gather the [purpose]s from [src1] and [src2] and put them on [dest].",
    "Go to [src1] and [src2] to collect the [purpose]s, then deliver them to [dest].",
    "Find a [purpose] at [src1] and another at [src2], then bring them both to [dest].",
    "Retrieve the [purpose]s from [src1] and [src2], and place them together on [dest].",
]

GATHER_FUZZY_TEMPLATES_3 = [
    "Pick up a [purpose] from each of [src1], [src2], and [src3], then place them all on [dest].",
    "Collect the [purpose]s from [src1], [src2], and [src3], and bring them to [dest].",
    "Gather all [purpose]s from [src1], [src2], and [src3] and put them on [dest].",
    "Find a [purpose] at [src1], [src2], and [src3], then deliver them all to [dest].",
]

FURNITURE_TYPE_DISPLAY = {
    "chestofdrawers": "chest of drawers",
    "teatable": "tea table",
    "shoecabinet": "shoe cabinet",
    "coffeetable": "coffee table",
    "tvstand": "TV stand",
    "nightstand": "nightstand",
}


# ── Shared utility functions ───────────────────────────────────────────────────

def _get_furniture_type(name: str) -> str:
    return name.rsplit('_', 1)[0]


def _normalize_room_name(room_name: str) -> str:
    return room_name.split('/')[0]


def _room_instance(room_name: str) -> int:
    parts = room_name.split('/')
    return int(parts[1]) if len(parts) > 1 else 0


def _is_articulated(furniture: Dict[str, Any]) -> bool:
    return 'door' in furniture.get('functionals', [])


def _furniture_index(name: str) -> int:
    """Extract the numeric suffix from a furniture name, e.g. 'desk_4' -> 4."""
    parts = name.rsplit('_', 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _furniture_type_display(name: str) -> str:
    ftype = _get_furniture_type(name)
    return FURNITURE_TYPE_DISPLAY.get(ftype, ftype)


def _fmt_obj_desc(caption: str) -> str:
    """Format detailed_caption as 'the <lowercased caption without leading article>'."""
    c = caption.rstrip('.').rstrip(',').strip()
    c = re.sub(r'^(a|an|the)\s+', '', c, flags=re.I)
    return f"the {c[0].lower()}{c[1:]}" if c else c


def _article(word: str) -> str:
    return "an" if word[0].lower() in 'aeiou' else "a"


def _ensure_category_in_caption(caption: str, category: str) -> str:
    """Append category to caption if not already present (case-insensitive)."""
    if category.lower() in caption.lower():
        return caption
    c = caption.rstrip('.').rstrip(',').strip()
    return f"{c}, {_article(category)} {category}"


# ── Placement verification (ported from scripts/filter/verify_placement.py) ────

UNSTABLE_FURNITURE_KEYWORDS = ('bed', 'couch', 'sofa', 'mattress', 'cushion', 'pillow')


def load_occ_map(occ_path) -> Optional[Dict]:
    """Load occupancy map from a .npy file. Returns None if not found."""
    if not _NUMPY_AVAILABLE:
        return None
    occ_path = Path(occ_path)
    if not occ_path.exists():
        return None
    occ_map = np.load(occ_path)
    x_coords = occ_map[0, 1:]
    y_coords = occ_map[1:, 0]
    free_map = occ_map[1:, 1:]
    x_res = float(np.mean(np.abs(np.diff(x_coords))))
    y_res = float(np.mean(np.abs(np.diff(y_coords))))
    return {
        'free_map': free_map,
        'x_origin': float(x_coords[0]),
        'y_origin': float(y_coords[0]),
        'x_res': x_res,
        'y_res': y_res,
        'H': free_map.shape[0],
        'W': free_map.shape[1],
    }


def _score_candidates_by_freeness(candidates, occ_data, radius=0.3, free_threshold=2):
    free_map = occ_data['free_map']
    x_orig, y_orig = occ_data['x_origin'], occ_data['y_origin']
    x_res, y_res = occ_data['x_res'], occ_data['y_res']
    H, W = occ_data['H'], occ_data['W']
    rx = max(1, int(np.ceil(radius / x_res)))
    ry = max(1, int(np.ceil(radius / y_res)))
    ixs = np.round((candidates[:, 0] - x_orig) / x_res).astype(int)
    iys = -np.round((candidates[:, 1] - y_orig) / y_res).astype(int)
    binary_free = (free_map >= free_threshold).astype(np.float32)
    scores = np.zeros(len(candidates), dtype=np.float32)
    for i in range(len(candidates)):
        cx, cy = int(ixs[i]), int(iys[i])
        y_lo, y_hi = max(0, cy - ry), min(H, cy + ry + 1)
        x_lo, x_hi = max(0, cx - rx), min(W, cx + rx + 1)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue
        scores[i] = binary_free[y_lo:y_hi, x_lo:x_hi].mean()
    return scores


def assign_positions_with_occmap(asset_list, receptacle_bbox, asset_sizes,
                                  occ_data=None, margin=0.05, grid_spacing=0.05,
                                  occ_radius=0.3) -> Optional[Dict]:
    """
    Grid-sampling placement with optional occupancy-map scoring.
    Returns {asset_id: (x, y, z)} or None if placement is impossible.
    """
    if not _NUMPY_AVAILABLE:
        return {}   # can't verify without numpy; treat as passed

    bbox_min = np.array(receptacle_bbox['min'])
    bbox_max = np.array(receptacle_bbox['max'])
    bbox_min, bbox_max = np.minimum(bbox_min, bbox_max), np.maximum(bbox_min, bbox_max)

    xs = np.arange(bbox_min[0] + grid_spacing / 2, bbox_max[0], grid_spacing)
    ys = np.arange(bbox_min[1] + grid_spacing / 2, bbox_max[1], grid_spacing)
    if len(xs) == 0 or len(ys) == 0:
        return None

    xx, yy = np.meshgrid(xs, ys)
    all_candidates = np.column_stack([xx.ravel(), yy.ravel()])

    if occ_data is not None:
        occ_scores = _score_candidates_by_freeness(all_candidates, occ_data, occ_radius)
    else:
        occ_scores = np.ones(len(all_candidates), dtype=np.float32)

    order = np.argsort(-occ_scores)
    all_candidates = all_candidates[order]
    occ_scores = occ_scores[order]

    placements = {}
    placed = []   # [(x, y, half_w, half_d)]

    for uid in asset_list:
        if uid not in asset_sizes:
            return None
        info = asset_sizes[uid]
        hw, hd, height = info['width'] / 2, info['depth'] / 2, info['height']

        valid = (
            (all_candidates[:, 0] >= bbox_min[0] + hw + margin) &
            (all_candidates[:, 0] <= bbox_max[0] - hw - margin) &
            (all_candidates[:, 1] >= bbox_min[1] + hd + margin) &
            (all_candidates[:, 1] <= bbox_max[1] - hd - margin)
        )
        for px, py, phw, phd in placed:
            valid &= ~(
                (np.abs(all_candidates[:, 0] - px) < hw + phw + margin) &
                (np.abs(all_candidates[:, 1] - py) < hd + phd + margin)
            )

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return None

        if placed:
            pa = np.array([(px, py) for px, py, _, _ in placed])
            min_dists = np.min(np.linalg.norm(
                all_candidates[valid_idx, None, :] - pa[None, :, :], axis=2), axis=1)
            max_d = np.max(min_dists)
            disp = min_dists / max_d if max_d > 0 else np.ones(len(valid_idx))
            combined = occ_scores[valid_idx] + disp
        else:
            combined = occ_scores[valid_idx]

        best = valid_idx[np.argmax(combined)]
        x, y = all_candidates[best]
        z = bbox_max[2] + height / 2 + 0.02
        placements[uid] = (float(x), float(y), float(z))
        placed.append((x, y, hw, hd))

    return placements


# ── Base Generator ─────────────────────────────────────────────────────────────

class BasePnPGenerator:
    """
    Shared base for all PnP generators.

    Handles data loading, index building, and common utilities.
    Subclasses implement generate_tasks_for_scene().
    """

    TASK_TYPE = "base"

    def __init__(
        self,
        furniture_library_path: str = 'assets/metadata',
        asset_library_path: str = 'assets/metadata/consolidated_asset_library_with_size.json',
        object_mapping_path: str = 'assets/metadata/furniture_pair_object_mapping.json',
        require_detailed_caption: bool = False,
        confidence_filter: bool = False,
        seed: Optional[int] = None,
        verify_placement: bool = False,
        occ_map_root: Optional[str] = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.furniture_library: Dict = self._load_furniture_library(furniture_library_path)
        self.asset_library: Dict = self._load(asset_library_path)
        self._resolve_usd_paths(self.asset_library)
        self.object_mapping: List = self._load(object_mapping_path)

        self._require_detailed_caption = require_detailed_caption
        self._confidence_filter = confidence_filter
        self._verify_placement = verify_placement
        self._occ_map_root = occ_map_root
        self._occ_cache: Dict = {}

        self._build_indices()
        self.stats: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _load(path) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _resolve_usd_paths(lib: Dict) -> None:
        """Prepend MESATASK_USD_ROOT to relative usd_path entries.

        The published consolidated_asset_library_with_size.json stores only the
        filename (e.g. ``abc123.usd``) so that users can place the MesaTask USD
        files anywhere locally.  Set the environment variable MESATASK_USD_ROOT
        to the directory containing those files before running any generator.
        """
        root = os.environ.get("MESATASK_USD_ROOT", "")
        if not root:
            return
        for entry in lib.values():
            path = entry.get("usd_path", "")
            if path and not os.path.isabs(path):
                entry["usd_path"] = os.path.join(root, path)

    @staticmethod
    def _load_furniture_library(path) -> Dict:
        """Load furniture library from a single JSON file or a metadata root directory.

        If *path* is a directory, it is assumed to be an `assets/metadata`-style
        root where each sub-directory contains a `scene_furniture_library.json`.
        All such files are merged into one dict (keys are furniture IDs).
        """
        p = Path(path)
        if p.is_file():
            with open(p, 'r') as f:
                return json.load(f)
        # Directory mode: merge every scene's library
        merged: Dict = {}
        for scene_dir in sorted(p.iterdir()):
            lib_file = scene_dir / 'scene_furniture_library.json'
            if lib_file.exists():
                with open(lib_file, 'r') as f:
                    merged.update(json.load(f))
        if not merged:
            raise FileNotFoundError(f"No scene_furniture_library.json found under {p}")
        return merged

    def _build_indices(self):
        # scene -> list of furniture
        self.scene_furniture: Dict[str, List[Dict]] = defaultdict(list)
        # (scene_id, name) -> furniture dict (scene-aware to avoid cross-scene collisions)
        self.furniture_by_scene_name: Dict[Tuple[str, str], Dict] = {}
        for _fid, fur in self.furniture_library.items():
            self.scene_furniture[fur['scene_id']].append(fur)
            self.furniture_by_scene_name[(fur['scene_id'], fur['name'])] = fur

        # category -> [uid]
        self.category_to_assets: Dict[str, List[str]] = defaultdict(list)
        for uid, asset in self.asset_library.items():
            if self._require_detailed_caption and not asset.get('detailed_caption'):
                continue
            primary = asset.get('category', '').lower()
            all_cats = asset.get('all_categories', {})
            if self._confidence_filter and all_cats:
                primary_count = (all_cats.get(asset.get('category', ''), 0)
                                 or max(all_cats.values()))
            else:
                primary_count = 1  # unused when confidence_filter=False
            if primary:
                self.category_to_assets[primary].append(uid)
            for cat, count in all_cats.items():
                cl = cat.lower()
                if cl == primary:
                    continue
                if self._confidence_filter and count < max(primary_count * 0.1, 3):
                    continue
                self.category_to_assets[cl].append(uid)

        # mapping lookups keyed by (src_room, src_type, dest_room, dest_type)
        # reverse_mapping_lookup stored under the SAME forward key; the generator
        # computes rev = (dest, src) which equals the original forward key.
        self.mapping_lookup: Dict[Tuple, List[str]] = {}
        self.reverse_mapping_lookup: Dict[Tuple, List[str]] = {}
        for m in self.object_mapping:
            key = (m['src']['room'].lower(), m['src']['furniture'].lower(),
                   m['dest']['room'].lower(), m['dest']['furniture'].lower())
            self.mapping_lookup[key] = m.get('objects_as_src', m['objects'])
            objs_dest = m.get('objects_as_dest', m['objects'])
            if objs_dest:
                self.reverse_mapping_lookup[key] = objs_dest

    # ── shared utilities ──────────────────────────────────────────────────────

    def _filter_valid_furniture(self, flist: List[Dict]) -> List[Dict]:
        return [f for f in flist if not _is_articulated(f)]

    def _get_furniture_distractors(self, target: str, all_names: List[str]) -> List[str]:
        tt = _get_furniture_type(target)
        return [n for n in all_names if n != target and _get_furniture_type(n) == tt]

    def _generate_caption_pairs(self, furniture_captions: Dict[str, List[str]]) -> List[Dict]:
        pairs = []
        items = list(furniture_captions.items())
        for (sn, sc), (dn, dc) in itertools.combinations(items, 2):
            for s, d in itertools.product(sc, dc):
                pairs.append({"src": {"name": sn, "caption": s},
                              "dest": {"name": dn, "caption": d}})
                pairs.append({"src": {"name": dn, "caption": d},
                              "dest": {"name": sn, "caption": s}})
        return pairs

    def _find_matching_assets(self, category: str) -> List[str]:
        cl = category.lower()
        if cl in self.category_to_assets:
            return self.category_to_assets[cl]
        for cat, uids in self.category_to_assets.items():
            if cl in cat or cat in cl:
                return uids
        return []

    def _get_mapping_categories(
        self, src_room: str, src_type: str, dest_room: str, dest_type: str
    ) -> Optional[List[str]]:
        key = (src_room.lower(), src_type.lower(), dest_room.lower(), dest_type.lower())
        cats = self.mapping_lookup.get(key)
        if cats:
            return cats
        rev = (dest_room.lower(), dest_type.lower(), src_room.lower(), src_type.lower())
        return self.reverse_mapping_lookup.get(rev)

    def _select_object_for_pair(
        self, src_room: str, src_type: str, dest_room: str, dest_type: str
    ) -> Optional[Tuple[str, str]]:
        cats = self._get_mapping_categories(src_room, src_type, dest_room, dest_type)
        if not cats:
            return None
        shuffled = cats.copy()
        random.shuffle(shuffled)
        for cat in shuffled:
            assets = self._find_matching_assets(cat)
            if assets:
                uid = random.choice(assets)
                primary_cat = self.asset_library.get(uid, {}).get('category', cat)
                return (primary_cat, uid)
        return None

    def _get_furniture_caption(self, furniture: Dict) -> str:
        caps = furniture.get('captions', [])
        cap = random.choice(caps) if caps else f"The {_furniture_type_display(furniture['name'])} in the room"
        return _ensure_category_in_caption(cap, _furniture_type_display(furniture['name']))

    def _get_scene_captions(self, valid_furniture: List[Dict]) -> Dict[str, List[str]]:
        result = {}
        for fur in valid_furniture:
            name = fur['name']
            caps = fur.get('captions', [])
            if not caps:
                caps = [f"The only {_get_furniture_type(name)} in the room."]
            ftype = _furniture_type_display(name)
            result[name] = [_ensure_category_in_caption(c, ftype) for c in caps]
        return result

    # ── execution plan helpers ────────────────────────────────────────────────

    def _nav_steps(self, distractors: List[str], target: str) -> List[Dict]:
        target_idx = _furniture_index(target)
        # Distractors with a smaller index than the target must be visited before
        # the target (they appear earlier in the agent's natural search order).
        # Larger-indexed distractors are shuffled in before the target as well,
        # since they also contain the target object in the initial world state.
        smaller = [d for d in distractors if _furniture_index(d) < target_idx]
        larger  = [d for d in distractors if _furniture_index(d) >= target_idx]
        random.shuffle(smaller)
        random.shuffle(larger)
        ordered = smaller + larger + [target]
        steps = []
        for fur in ordered:
            steps.append({"action": "nav_to", "args": {"receptacle_name": fur}})
            if fur == target:
                break
        return steps

    def _pick_steps(self, category: str, description: Optional[str] = None) -> List[Dict]:
        args: Dict[str, Any] = {"target_category": category}
        if description:
            args["target_description"] = description
        return [
            {"action": "find",    "args": dict(args)},
            {"action": "gaze_at", "args": dict(args)},
            {"action": "pick",    "args": dict(args)},
        ]

    def _place_step(self, receptacle: str) -> Dict:
        return {"action": "place", "args": {"receptacle_name": receptacle}}

    def _open_step(self, receptacle: str) -> Dict:
        return {"action": "open", "args": {"receptacle_name": receptacle}}

    def _close_step(self, receptacle: str) -> Dict:
        return {"action": "close", "args": {"receptacle_name": receptacle}}

    def _ask_step(self, category: str, description: Optional[str] = None) -> Dict:
        args: Dict[str, Any] = {"target_category": category}
        if description:
            args["target_description"] = description
        return {"action": "ask", "args": args}

    # ── description helpers ───────────────────────────────────────────────────

    def _desc_basic(self, category: str, src_cap: str, dest_cap: str) -> str:
        t = random.choice(STYLE_TEMPLATES)
        obj = f"{_article(category)} {category}"
        return (t.replace("[obj]", obj)
                 .replace("[src]", src_cap.rstrip('.'))
                 .replace("[dest]", dest_cap.rstrip('.')))

    def _desc_detailed(self, detailed_caption: str, src_cap: str, dest_cap: str) -> str:
        t = random.choice(STYLE_TEMPLATES_DESC)
        return (t.replace("[obj_desc]", _fmt_obj_desc(detailed_caption))
                 .replace("[src]", src_cap.rstrip('.'))
                 .replace("[dest]", dest_cap.rstrip('.')))

    def _desc_fuzzy(self, purpose: str, src_cap: str, dest_cap: str) -> str:
        t = random.choice(FUZZY_STYLE_TEMPLATES)
        return (t.replace("[purpose]", purpose)
                 .replace("[src]", src_cap.rstrip('.'))
                 .replace("[dest]", dest_cap.rstrip('.')))

    # ── common interface ──────────────────────────────────────────────────────

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        raise NotImplementedError

    def generate_tasks(self, scene_ids: Optional[List[str]] = None) -> List[Dict]:
        if scene_ids is None:
            scene_ids = list(self.scene_furniture.keys())
        all_tasks = []
        for sid in scene_ids:
            all_tasks.extend(self.generate_tasks_for_scene(sid))
        return all_tasks

    def save_tasks(self, tasks: List[Dict], output_path: str):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(tasks)} {self.TASK_TYPE} tasks to {p}")

    def _scene_setup(self, scene_id: str):
        """Common setup: load and filter furniture for a scene."""
        furniture_list = self.scene_furniture.get(scene_id, [])
        if not furniture_list:
            return None, None, None, None
        valid = self._filter_valid_furniture(furniture_list)
        if len(valid) < 2:
            return None, None, None, None
        self.stats['pairs_skipped_door_furniture'] += len(furniture_list) - len(valid)
        fur_by_name = {f['name']: f for f in valid}
        captions = self._get_scene_captions(valid)
        return furniture_list, valid, fur_by_name, captions

    # ── placement verification helpers ───────────────────────────────────────

    def _load_occ_data(self, scene_id: str):
        if scene_id not in self._occ_cache:
            if self._occ_map_root:
                occ_path = Path(self._occ_map_root) / scene_id / "occupancy.npy"
                self._occ_cache[scene_id] = load_occ_map(occ_path)
            else:
                self._occ_cache[scene_id] = None
        return self._occ_cache[scene_id]

    def _build_asset_sizes(self, uids: List[str]) -> Optional[Dict]:
        sizes = {}
        for uid in uids:
            if uid is None:
                return None
            asset = self.asset_library.get(uid)
            if asset is None:
                return None
            size = asset.get('size')
            if not size or len(size) < 3:
                return None
            sizes[uid] = {
                'width': size[0], 'depth': size[1], 'height': size[2],
                'radius': (size[0] ** 2 + size[1] ** 2) ** 0.5 / 2,
            }
        return sizes

    def _can_place_on(self, scene_id: str, fur_name: str, uids: List[str], occ_data) -> bool:
        """Check whether all uids can be placed on the given furniture."""
        if any(kw in fur_name.lower() for kw in UNSTABLE_FURNITURE_KEYWORDS):
            return False
        fur = self.furniture_by_scene_name.get((scene_id, fur_name))
        if fur is None:
            return False
        bbox = fur.get('receptacle_bbox')
        if not bbox:
            return False
        asset_sizes = self._build_asset_sizes(uids)
        if asset_sizes is None:
            return False
        return assign_positions_with_occmap(
            asset_list=uids, receptacle_bbox=bbox,
            asset_sizes=asset_sizes, occ_data=occ_data,
        ) is not None

    def _check_layout(self, task: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Verify that:
          1. Each furniture in initial_world_graph with content can physically fit
             all assigned objects (covers: src with target+distractors, and
             src_distractors for articulation tasks which appear in the graph).
          2. Each src_distractor furniture (same-type furniture the robot may visit
             first) can fit the target object alone, so confusion is plausible.

        Returns (passed, computed_placements).
        computed_placements maps "{uid}_on_{fur_name}" -> placement info dict,
        matching the format produced by batch_static.sh / verify_placement.py.
        """
        if not self._verify_placement:
            return True, None

        scene_id = task['scene_id']
        occ_data = self._load_occ_data(scene_id)
        all_placements: Dict = {}

        # ── 1. Check initial_world_graph content placements ──────────────────
        for fur_name, payload in task.get('initial_world_graph', {}).items():
            content = [uid for uid in payload.get('content', []) if uid is not None]
            if not content:
                continue
            if any(kw in fur_name.lower() for kw in UNSTABLE_FURNITURE_KEYWORDS):
                self.stats['placement_skipped_unstable'] += 1
                return False, None
            fur = self.furniture_by_scene_name.get((scene_id, fur_name))
            if fur is None:
                return False, None
            bbox = fur.get('receptacle_bbox')
            if not bbox:
                return False, None
            asset_sizes = self._build_asset_sizes(content)
            if asset_sizes is None:
                return False, None
            placements = assign_positions_with_occmap(
                asset_list=content, receptacle_bbox=bbox,
                asset_sizes=asset_sizes, occ_data=occ_data,
            )
            if placements is None:
                return False, None
            for uid, pos in placements.items():
                all_placements[f"{uid}_on_{fur_name}"] = {
                    'position': pos, 'furniture': fur_name, 'original_id': uid,
                }

        # ── 2. Check src_distractor furniture can hold the target alone ───────
        # (These are same-type furniture distractors not present in world_graph.)
        target_uids = list(task.get('obj_meta', {}).keys())

        # Simple task types: basic / distractor / interactive / articulation
        src_distractors = task.get('src_distractors', [])
        if src_distractors and target_uids:
            target_uid = target_uids[0]
            for fur_name in src_distractors:
                if not self._can_place_on(scene_id, fur_name, [target_uid], occ_data):
                    return False, None

        # Gather task type: per-source furniture distractors
        for src_info in task.get('sources', []):
            src_name = src_info['name']
            target_uid = src_info['object_uid']
            distractors = task.get('src_furniture_distractors', {}).get(src_name, [])
            for fur_name in distractors:
                if not self._can_place_on(scene_id, fur_name, [target_uid], occ_data):
                    return False, None

        return True, all_placements or None


# ── BasicPnPGenerator ─────────────────────────────────────────────────────────

class BasicPnPGenerator(BasePnPGenerator):
    """Simple pick-and-place with furniture distractors."""

    TASK_TYPE = "basic"

    def __init__(self, **kwargs):
        super().__init__(require_detailed_caption=False, confidence_filter=False, **kwargs)

    def _build_world_graph(self, furniture_list, src_name, dest_name, asset_uid, is_initial):
        wg = {}
        for fur in furniture_list:
            name = fur['name']
            entry = {"room": fur.get('room_name', ''), "content": []}
            if is_initial and name == src_name:
                entry["content"] = [asset_uid]
            elif not is_initial and name == dest_name:
                entry["content"] = [asset_uid]
            wg[name] = entry
        return wg

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        furniture_list, valid, fur_by_name, captions = self._scene_setup(scene_id)
        if valid is None:
            return []

        tasks = []
        generated_descs: set = set()
        all_names = [f['name'] for f in valid]
        pairs = self._generate_caption_pairs(captions)

        for pair in pairs:
            sn, sc = pair['src']['name'], pair['src']['caption']
            dn, dc = pair['dest']['name'], pair['dest']['caption']
            sf, df = fur_by_name[sn], fur_by_name[dn]

            result = self._select_object_for_pair(
                _normalize_room_name(sf.get('room_name', '')), _get_furniture_type(sn),
                _normalize_room_name(df.get('room_name', '')), _get_furniture_type(dn))
            if result is None:
                self.stats['pairs_skipped_no_mapping'] += 1
                continue

            cat, uid = result
            asset = self.asset_library.get(uid)
            if asset is None:
                self.stats['pairs_skipped_no_assets'] += 1
                continue

            desc = self._desc_basic(cat, sc, dc)
            if desc in generated_descs:
                self.stats['duplicate_descriptions_skipped'] += 1
                continue
            generated_descs.add(desc)

            src_fd = self._get_furniture_distractors(sn, all_names)
            dest_fd = self._get_furniture_distractors(dn, all_names)
            plan = (self._nav_steps(src_fd, sn) + self._pick_steps(cat) +
                    self._nav_steps(dest_fd, dn) + [self._place_step(dn)])

            task = {
                "scene_id": scene_id,
                "task_type": self.TASK_TYPE,
                "task_description": desc,
                "initial_world_graph": self._build_world_graph(furniture_list, sn, dn, uid, True),
                "goal_world_graph": self._build_world_graph(furniture_list, sn, dn, uid, False),
                "src": sn, "src_distractors": src_fd,
                "dest": dn, "dest_distractors": dest_fd,
                "obj_meta": {uid: {"category": cat,
                                   "usd_path": asset.get('usd_path', ''),
                                   "size": asset.get('size', [])}},
                "obj_distractors": [],
                "execution_plan": plan,
            }
            passed, placements = self._check_layout(task)
            if not passed:
                self.stats['placement_check_failed'] += 1
                continue
            if placements:
                task['computed_placements'] = placements
            tasks.append(task)
            self.stats['tasks_generated'] += 1

        self.stats['scenes_processed'] += 1
        return tasks


# ── DistractorPnPGenerator ────────────────────────────────────────────────────

class DistractorPnPGenerator(BasePnPGenerator):
    """Pick-and-place with same-category object distractors; uses detailed_caption."""

    TASK_TYPE = "distractor"

    def __init__(self, num_distractors: int = 2, **kwargs):
        self.num_distractors = num_distractors
        super().__init__(require_detailed_caption=True, confidence_filter=False, **kwargs)

    def _select_objects_for_pair(self, src_room, src_type, dest_room, dest_type):
        cats = self._get_mapping_categories(src_room, src_type, dest_room, dest_type)
        if not cats:
            return None
        needed = 1 + self.num_distractors
        shuffled = cats.copy()
        random.shuffle(shuffled)
        for cat in shuffled:
            assets = self._find_matching_assets(cat)
            if len(assets) >= needed:
                selected = random.sample(assets, needed)
                primary_cat = self.asset_library.get(selected[0], {}).get('category', cat)
                return primary_cat, selected[0], selected[1:]
        return None

    def _build_world_graph(self, furniture_list, src_name, dest_name,
                           target_uid, distractor_uids, is_initial, src_distractors=None):
        src_dist_set = set(src_distractors) if src_distractors else set()
        wg = {}
        for fur in furniture_list:
            name = fur['name']
            entry = {"room": fur.get('room_name', ''), "content": []}
            if is_initial and name == src_name:
                entry["content"] = [target_uid] + list(distractor_uids)
            elif is_initial and name in src_dist_set:
                entry["content"] = [target_uid]
            elif not is_initial:
                if name == src_name:
                    entry["content"] = list(distractor_uids)
                elif name == dest_name:
                    entry["content"] = [target_uid]
                elif name in src_dist_set:
                    entry["content"] = [target_uid]
            wg[name] = entry
        return wg

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        furniture_list, valid, fur_by_name, captions = self._scene_setup(scene_id)
        if valid is None:
            return []

        tasks = []
        generated_descs: set = set()
        all_names = [f['name'] for f in valid]
        pairs = self._generate_caption_pairs(captions)

        for pair in pairs:
            sn, sc = pair['src']['name'], pair['src']['caption']
            dn, dc = pair['dest']['name'], pair['dest']['caption']
            sf, df = fur_by_name[sn], fur_by_name[dn]

            result = self._select_objects_for_pair(
                _normalize_room_name(sf.get('room_name', '')), _get_furniture_type(sn),
                _normalize_room_name(df.get('room_name', '')), _get_furniture_type(dn))
            if result is None:
                self.stats['pairs_skipped_no_mapping'] += 1
                continue

            cat, target_uid, dist_uids = result
            target_asset = self.asset_library.get(target_uid)
            if target_asset is None:
                self.stats['pairs_skipped_no_assets'] += 1
                continue

            cap = target_asset.get('detailed_caption', '')
            if not cap:
                self.stats['pairs_skipped_no_detailed_caption'] += 1
                continue
            cap = _ensure_category_in_caption(cap, cat)

            desc = self._desc_detailed(cap, sc, dc)
            if desc in generated_descs:
                self.stats['duplicate_descriptions_skipped'] += 1
                continue
            generated_descs.add(desc)

            src_fd = self._get_furniture_distractors(sn, all_names)
            dest_fd = self._get_furniture_distractors(dn, all_names)
            plan = (self._nav_steps(src_fd, sn) + self._pick_steps(cat, cap) +
                    self._nav_steps(dest_fd, dn) + [self._place_step(dn)])

            dist_meta = {}
            for duid in dist_uids:
                da = self.asset_library.get(duid, {})
                dist_meta[duid] = {
                    "category": da.get('category', cat),
                    "detailed_caption": da.get('detailed_caption', ''),
                    "usd_path": da.get('usd_path', ''),
                    "size": da.get('size', []),
                }

            task = {
                "scene_id": scene_id,
                "task_type": self.TASK_TYPE,
                "task_description": desc,
                "initial_world_graph": self._build_world_graph(
                    furniture_list, sn, dn, target_uid, dist_uids, True, src_fd),
                "goal_world_graph": self._build_world_graph(
                    furniture_list, sn, dn, target_uid, dist_uids, False, src_fd),
                "src": sn, "src_distractors": src_fd,
                "dest": dn, "dest_distractors": dest_fd,
                "obj_meta": {target_uid: {
                    "category": cat, "detailed_caption": cap,
                    "usd_path": target_asset.get('usd_path', ''),
                    "size": target_asset.get('size', []),
                }},
                "obj_distractors": dist_uids,
                "obj_distractor_meta": dist_meta,
                "execution_plan": plan,
            }
            passed, placements = self._check_layout(task)
            if not passed:
                self.stats['placement_check_failed'] += 1
                continue
            if placements:
                task['computed_placements'] = placements
            tasks.append(task)
            self.stats['tasks_generated'] += 1

        self.stats['scenes_processed'] += 1
        return tasks


# ── ArticulationPnPGenerator ──────────────────────────────────────────────────

class ArticulationPnPGenerator(BasePnPGenerator):
    """Pick-and-place involving articulated furniture (open/close door)."""

    TASK_TYPE = "articulation"

    def __init__(self, max_objects_per_pair: int = 5, num_distractors: int = 2, **kwargs):
        self.max_objects_per_pair = max_objects_per_pair
        self.num_distractors = num_distractors
        super().__init__(require_detailed_caption=False, confidence_filter=False, **kwargs)

    def _get_objects_for_pair(self, src_room, src_type, dest_room, dest_type, count):
        cats = self._get_mapping_categories(src_room, src_type, dest_room, dest_type)
        if not cats:
            return []
        shuffled = cats.copy()
        random.shuffle(shuffled)
        results, seen_cats = [], set()
        for cat in shuffled:
            if len(results) >= count:
                break
            if cat in seen_cats:
                continue
            assets = self._find_matching_assets(cat)
            if assets:
                uid = random.choice(assets)
                primary_cat = self.asset_library.get(uid, {}).get('category', cat)
                results.append((primary_cat, uid))
                seen_cats.add(cat)
        return results

    def _get_distractors_non_artic(self, target: str, furniture_list: List[Dict]) -> List[str]:
        tt = _get_furniture_type(target)
        return [f['name'] for f in furniture_list
                if f['name'] != target
                and _get_furniture_type(f['name']) == tt
                and not _is_articulated(f)]

    def _select_obj_distractors(self, cat: str, target_uid: str) -> List[str]:
        """Return up to num_distractors same-category object UIDs excluding target."""
        assets = self._find_matching_assets(cat)
        candidates = [a for a in assets if a != target_uid]
        if len(candidates) <= self.num_distractors:
            return candidates
        return random.sample(candidates, self.num_distractors)

    def _build_world_graph(self, furniture_list, src_name, dest_name, asset_uid,
                           src_distractors=None, is_initial=True, obj_dist_uids=None):
        src_dist_set = set(src_distractors) if src_distractors else set()
        obj_dist_list = list(obj_dist_uids) if obj_dist_uids else []
        wg = {}
        for fur in furniture_list:
            name = fur['name']
            entry = {"room": fur.get('room_name', ''), "content": []}
            if _is_articulated(fur):
                entry["door"] = False
            if is_initial:
                if name == src_name:
                    entry["content"] = [asset_uid] + obj_dist_list
                elif name in src_dist_set:
                    entry["content"] = [asset_uid]
            elif not is_initial:
                if name == src_name:
                    entry["content"] = obj_dist_list
                elif name == dest_name:
                    entry["content"] = [asset_uid]
                elif name in src_dist_set:
                    entry["content"] = [asset_uid]
            wg[name] = entry
        return wg

    def _store_plan(self, src_name, dest_name, src_distractors, category):
        """normal → articulated: open dest, pick from src, place in dest, close dest."""
        plan = [{"action": "nav_to", "args": {"receptacle_name": dest_name}},
                self._open_step(dest_name)]
        plan += self._nav_steps(src_distractors, src_name)
        plan += self._pick_steps(category)
        plan += [{"action": "nav_to", "args": {"receptacle_name": dest_name}},
                 self._place_step(dest_name),
                 self._close_step(dest_name)]
        return plan

    def _retrieve_plan(self, src_name, dest_name, dest_distractors, category):
        """articulated → normal: open src, pick, place at dest, close src."""
        plan = [{"action": "nav_to", "args": {"receptacle_name": src_name}},
                self._open_step(src_name)]
        plan += self._pick_steps(category)
        plan += self._nav_steps(dest_distractors, dest_name)
        plan += [self._place_step(dest_name),
                 {"action": "nav_to", "args": {"receptacle_name": src_name}},
                 self._close_step(src_name)]
        return plan

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        furniture_list = self.scene_furniture.get(scene_id, [])
        if not furniture_list:
            return []

        artic = [f for f in furniture_list if _is_articulated(f)]
        normal = [f for f in furniture_list if not _is_articulated(f)]
        if not artic or not normal:
            return []

        tasks = []
        generated_descs: set = set()

        def _make_task(sf, df, task_type, plan_fn, src_dist):
            sr = _normalize_room_name(sf.get('room_name', ''))
            st = _get_furniture_type(sf['name'])
            dr = _normalize_room_name(df.get('room_name', ''))
            dt = _get_furniture_type(df['name'])
            objects = self._get_objects_for_pair(sr, st, dr, dt, self.max_objects_per_pair)
            if not objects:
                self.stats['pairs_skipped_no_mapping'] += 1
                return []
            result = []
            for cat, uid in objects:
                asset = self.asset_library.get(uid)
                if asset is None:
                    self.stats['pairs_skipped_no_assets'] += 1
                    continue
                sc = self._get_furniture_caption(sf)
                dc = self._get_furniture_caption(df)
                desc = self._desc_basic(cat, sc, dc)
                if desc in generated_descs:
                    self.stats['duplicate_descriptions_skipped'] += 1
                    continue
                generated_descs.add(desc)
                distractors = (self._get_distractors_non_artic(sf['name'], normal)
                               if src_dist else [])
                dest_dist = (self._get_distractors_non_artic(df['name'], normal)
                             if not src_dist else [])
                obj_dist_uids = self._select_obj_distractors(cat, uid)
                obj_dist_meta = {}
                for duid in obj_dist_uids:
                    da = self.asset_library.get(duid, {})
                    obj_dist_meta[duid] = {
                        "category": cat,
                        "usd_path": da.get('usd_path', ''),
                        "size": da.get('size', []),
                    }
                plan = plan_fn(sf['name'], df['name'],
                               distractors if src_dist else dest_dist, cat)
                task = {
                    "scene_id": scene_id,
                    "task_type": task_type,
                    "task_description": desc,
                    "initial_world_graph": self._build_world_graph(
                        furniture_list, sf['name'], df['name'], uid,
                        distractors if src_dist else None, True, obj_dist_uids),
                    "goal_world_graph": self._build_world_graph(
                        furniture_list, sf['name'], df['name'], uid,
                        distractors if src_dist else None, False, obj_dist_uids),
                    "src": sf['name'],
                    "src_distractors": distractors if src_dist else [],
                    "dest": df['name'],
                    "dest_distractors": dest_dist,
                    "obj_meta": {uid: {"category": cat,
                                       "usd_path": asset.get('usd_path', ''),
                                       "size": asset.get('size', [])}},
                    "obj_distractors": obj_dist_uids,
                    "obj_distractor_meta": obj_dist_meta,
                    "execution_plan": plan,
                }
                passed, placements = self._check_layout(task)
                if not passed:
                    self.stats['placement_check_failed'] += 1
                    continue
                if placements:
                    task['computed_placements'] = placements
                result.append(task)
            return result

        for nf in normal:
            for af in artic:
                for t in _make_task(nf, af, "store", self._store_plan, src_dist=True):
                    tasks.append(t)
                    self.stats['store_tasks_generated'] += 1

        for af in artic:
            for nf in normal:
                for t in _make_task(af, nf, "retrieve", self._retrieve_plan, src_dist=False):
                    tasks.append(t)
                    self.stats['retrieve_tasks_generated'] += 1

        self.stats['scenes_processed'] += 1
        return tasks


# ── InteractivePnPGenerator ──────────────────────────────────────────────────

class InteractivePnPGenerator(BasePnPGenerator):
    """Pick-and-place with same-purpose different-category distractors + fuzzy description."""

    TASK_TYPE = "interactive"

    def __init__(self, category_pairs_path: str = 'assets/metadata/category_pairs_same_purpose.json',
                 num_distractors: int = 2, **kwargs):
        self.num_distractors = num_distractors
        super().__init__(require_detailed_caption=True, confidence_filter=True, **kwargs)
        self.purpose_groups: Dict = self._load(category_pairs_path)
        self._build_purpose_index()

    def _build_purpose_index(self):
        self.category_to_purpose: Dict[str, Dict] = {}
        for gk, gd in self.purpose_groups.items():
            for cat in gd.get('categories', []):
                self.category_to_purpose[cat.lower()] = {
                    'group_key': gk, 'purpose': gd['purpose']}

    def _select_objects_for_pair(self, src_room, src_type, dest_room, dest_type):
        cats = self._get_mapping_categories(src_room, src_type, dest_room, dest_type)
        if not cats:
            return None
        shuffled = cats.copy()
        random.shuffle(shuffled)

        for cat in shuffled:
            cl = cat.lower()
            purpose_info = self.category_to_purpose.get(cl)
            if not purpose_info:
                continue
            target_assets = self._find_matching_assets(cat)
            if not target_assets:
                continue

            gk = purpose_info['group_key']
            gd = self.purpose_groups[gk]

            # Recommended partners first, then rest of group
            recommended = []
            for pair in gd.get('recommended_pairs', []):
                pl = [p.lower() for p in pair]
                if cl in pl:
                    partner = pl[1] if pl[0] == cl else pl[0]
                    if partner not in recommended:
                        recommended.append(partner)
            ordered_alt = list(recommended)
            for c in [c.lower() for c in gd['categories'] if c.lower() != cl]:
                if c not in ordered_alt:
                    ordered_alt.append(c)

            target_set = set(target_assets)
            pool: List[Tuple[str, str]] = []
            for alt_cat in ordered_alt:
                for uid in self._find_matching_assets(alt_cat):
                    if uid not in target_set:
                        pool.append((uid, alt_cat))

            if len(pool) < self.num_distractors:
                continue

            target_uid = random.choice(target_assets)
            primary_cat = self.asset_library.get(target_uid, {}).get('category', cat)
            random.shuffle(pool)
            selected: List[Tuple[str, str]] = []
            seen_uids = {target_uid}
            for uid, dc in pool:
                if uid not in seen_uids:
                    dc_primary = self.asset_library.get(uid, {}).get('category', dc)
                    selected.append((uid, dc_primary))
                    seen_uids.add(uid)
                if len(selected) >= self.num_distractors:
                    break

            if len(selected) < self.num_distractors:
                continue

            return (primary_cat, target_uid,
                    [u for u, _ in selected],
                    [c for _, c in selected],
                    gd['purpose'], gk)
        return None

    def _build_world_graph(self, furniture_list, src_name, dest_name,
                           target_uid, distractor_uids, is_initial, src_distractors=None):
        src_dist_set = set(src_distractors) if src_distractors else set()
        wg = {}
        for fur in furniture_list:
            name = fur['name']
            entry = {"room": fur.get('room_name', ''), "content": []}
            if is_initial:
                if name == src_name:
                    entry["content"] = [target_uid] + list(distractor_uids)
                elif name in src_dist_set:
                    entry["content"] = [target_uid]
            elif not is_initial:
                if name == src_name:
                    entry["content"] = list(distractor_uids)
                elif name == dest_name:
                    entry["content"] = [target_uid]
                elif name in src_dist_set:
                    entry["content"] = [target_uid]
            wg[name] = entry
        return wg

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        furniture_list, valid, fur_by_name, captions = self._scene_setup(scene_id)
        if valid is None:
            return []

        tasks = []
        generated_descs: set = set()
        all_names = [f['name'] for f in valid]
        pairs = self._generate_caption_pairs(captions)

        for pair in pairs:
            sn, sc = pair['src']['name'], pair['src']['caption']
            dn, dc = pair['dest']['name'], pair['dest']['caption']
            sf, df = fur_by_name[sn], fur_by_name[dn]

            result = self._select_objects_for_pair(
                _normalize_room_name(sf.get('room_name', '')), _get_furniture_type(sn),
                _normalize_room_name(df.get('room_name', '')), _get_furniture_type(dn))
            if result is None:
                self.stats['pairs_skipped_no_mapping'] += 1
                continue

            cat, target_uid, dist_uids, dist_cats, purpose, gk = result
            target_asset = self.asset_library.get(target_uid)
            if target_asset is None:
                self.stats['pairs_skipped_no_assets'] += 1
                continue

            cap = target_asset.get('detailed_caption', '')
            if not cap:
                self.stats['pairs_skipped_no_detailed_caption'] += 1
                continue
            cap = _ensure_category_in_caption(cap, cat)

            fuzzy_desc = self._desc_fuzzy(purpose, sc, dc)
            detailed_desc = self._desc_detailed(cap, sc, dc)
            if fuzzy_desc in generated_descs:
                self.stats['duplicate_descriptions_skipped'] += 1
                continue
            generated_descs.add(fuzzy_desc)

            src_fd = self._get_furniture_distractors(sn, all_names)
            dest_fd = self._get_furniture_distractors(dn, all_names)
            plan = (self._nav_steps(src_fd, sn) +
                    [self._ask_step(cat, cap)] +
                    self._pick_steps(cat, cap) +
                    self._nav_steps(dest_fd, dn) + [self._place_step(dn)])

            dist_meta = {}
            for duid, dcat in zip(dist_uids, dist_cats):
                da = self.asset_library.get(duid, {})
                dist_meta[duid] = {
                    "category": dcat,
                    "detailed_caption": da.get('detailed_caption', ''),
                    "usd_path": da.get('usd_path', ''),
                    "size": da.get('size', []),
                }

            task = {
                "scene_id": scene_id,
                "task_type": self.TASK_TYPE,
                "task_description": fuzzy_desc,
                "detailed_task_description": detailed_desc,
                "purpose": purpose,
                "purpose_group": gk,
                "initial_world_graph": self._build_world_graph(
                    furniture_list, sn, dn, target_uid, dist_uids, True, src_fd),
                "goal_world_graph": self._build_world_graph(
                    furniture_list, sn, dn, target_uid, dist_uids, False, src_fd),
                "src": sn, "src_distractors": src_fd,
                "dest": dn, "dest_distractors": dest_fd,
                "obj_meta": {target_uid: {
                    "category": cat, "detailed_caption": cap,
                    "usd_path": target_asset.get('usd_path', ''),
                    "size": target_asset.get('size', []),
                }},
                "obj_distractors": dist_uids,
                "obj_distractor_meta": dist_meta,
                "execution_plan": plan,
            }
            passed, placements = self._check_layout(task)
            if not passed:
                self.stats['placement_check_failed'] += 1
                continue
            if placements:
                task['computed_placements'] = placements
            tasks.append(task)
            self.stats['tasks_generated'] += 1

        self.stats['scenes_processed'] += 1
        return tasks


# ── GatherPnPGenerator ────────────────────────────────────────────────────────

class GatherPnPGenerator(BasePnPGenerator):
    """Multi-source gather: collect N same-purpose objects to one destination."""

    TASK_TYPE = "gather"

    def __init__(
        self,
        category_pairs_path: str = 'assets/metadata/category_pairs_same_purpose.json',
        num_objects: int = 2,
        num_same_cat_distractors: int = 1,
        num_diff_cat_distractors: int = 1,
        max_purposes_per_combo: int = 3,
        **kwargs
    ):
        assert num_objects in (2, 3), "num_objects must be 2 or 3"
        self.num_objects = num_objects
        self.num_same_cat_distractors = num_same_cat_distractors
        self.num_diff_cat_distractors = num_diff_cat_distractors
        self.max_purposes_per_combo = max_purposes_per_combo

        super().__init__(require_detailed_caption=True, confidence_filter=True, **kwargs)
        self.purpose_groups: Dict = self._load(category_pairs_path)
        self._build_purpose_indices()

    def _build_purpose_indices(self):
        self.category_to_purpose: Dict[str, Dict] = {}
        for gk, gd in self.purpose_groups.items():
            for cat in gd.get('categories', []):
                self.category_to_purpose[cat.lower()] = {
                    'group_key': gk, 'purpose': gd['purpose']}

        all_asset_cats = set(self.category_to_assets.keys())
        all_purpose_cats: set = set()
        for gd in self.purpose_groups.values():
            for c in gd.get('categories', []):
                all_purpose_cats.add(c.lower())

        self.purpose_cat_to_asset_cats: Dict[str, set] = {}
        self.cat_to_purpose_cats: Dict[str, set] = defaultdict(set)
        for pcat in all_purpose_cats:
            matches: set = {acat for acat in all_asset_cats if self._cats_match(pcat, acat)}
            self.purpose_cat_to_asset_cats[pcat] = matches
            for m in matches:
                self.cat_to_purpose_cats[m].add(pcat)

    @staticmethod
    def _cats_match(a: str, b: str) -> bool:
        if a == b:
            return True
        return bool(set(a.split()) & set(b.split()))

    def _find_assets_gather(self, category: str) -> List[str]:
        cl = category.lower()
        seen: set = set()
        result: List[str] = []

        def _add(uids):
            for u in uids:
                if u not in seen:
                    result.append(u)
                    seen.add(u)

        expanded = self.purpose_cat_to_asset_cats.get(cl)
        if expanded:
            for acat in expanded:
                _add(self.category_to_assets.get(acat, []))
        if cl in self.category_to_assets:
            _add(self.category_to_assets[cl])
        if not result:
            for cat, uids in self.category_to_assets.items():
                if self._cats_match(cl, cat):
                    _add(uids)
        return result

    def _count_room_instances(self, scene_id: str) -> Dict[str, int]:
        rooms = {f.get('room_name', '') for f in self.scene_furniture.get(scene_id, [])}
        counts: Dict[str, int] = defaultdict(int)
        for r in rooms:
            counts[_normalize_room_name(r)] += 1
        return counts

    def _format_src_ref(self, furniture: Dict, room_counts: Dict[str, int]) -> str:
        ftype = _furniture_type_display(furniture['name'])
        room_name = furniture.get('room_name', '')
        room_type = _normalize_room_name(room_name)
        if room_counts.get(room_type, 1) <= 1:
            room_ref = f"the {room_type}"
        else:
            room_ref = f"{room_type} {_room_instance(room_name) + 1}"
        return f"the {ftype} in {room_ref}"

    def _find_same_room_groups(self, scene_id: str):
        flist = self._filter_valid_furniture(self.scene_furniture.get(scene_id, []))
        room_map: Dict[str, List[Dict]] = defaultdict(list)
        for f in flist:
            room_map[f['room_name']].append(f)
        needed = self.num_objects + 1
        for room, items in room_map.items():
            if len(items) >= needed:
                yield room, items

    def _find_cross_room_groups(self, scene_id: str):
        flist = self._filter_valid_furniture(self.scene_furniture.get(scene_id, []))
        type_map: Dict[str, List[Dict]] = defaultdict(list)
        for f in flist:
            type_map[_get_furniture_type(f['name'])].append(f)
        for ftype, items in type_map.items():
            rooms = {f['room_name'] for f in items}
            if len(rooms) >= self.num_objects:
                yield ftype, items

    def _select_all_purpose_objects(self, src_infos, dest_info):
        dest_room = _normalize_room_name(dest_info.get('room_name', ''))
        dest_type = _get_furniture_type(dest_info['name'])
        per_src_cats = [
            self._get_mapping_categories(
                _normalize_room_name(si.get('room_name', '')),
                _get_furniture_type(si['name']),
                dest_room, dest_type)
            for si in src_infos
        ]

        group_keys = list(self.purpose_groups.keys())
        random.shuffle(group_keys)
        results = []
        used_gks: set = set()

        for gk in group_keys:
            if len(results) >= self.max_purposes_per_combo:
                break
            gd = self.purpose_groups[gk]
            group_cats = [c.lower() for c in gd['categories']]
            group_cats_set = set(group_cats)
            expanded_group: set = set()
            for gc in group_cats:
                expanded_group.update(self.purpose_cat_to_asset_cats.get(gc, {gc}))

            src_candidate_cats: List[List[str]] = []
            for i in range(len(src_infos)):
                if per_src_cats[i] is not None:
                    valid: List[str] = []
                    seen_v: set = set()
                    for c in per_src_cats[i]:
                        cl = c.lower()
                        if cl in group_cats_set and cl not in seen_v:
                            valid.append(cl); seen_v.add(cl)
                        elif cl in expanded_group and cl not in seen_v:
                            valid.append(cl); seen_v.add(cl)
                        else:
                            for gc in group_cats:
                                if self._cats_match(cl, gc) and gc not in seen_v:
                                    valid.append(gc); seen_v.add(gc); break
                else:
                    valid = list(group_cats)
                if not valid:
                    break
                src_candidate_cats.append(valid)
            else:
                r = self._assign_objects(src_candidate_cats, gd)
                if r:
                    results.append((gd['purpose'], gk, r))
                    used_gks.add(gk)

        if not results:
            for gk in group_keys:
                if gk in used_gks or len(results) >= self.max_purposes_per_combo:
                    continue
                gd = self.purpose_groups[gk]
                src_cands = [[c.lower() for c in gd['categories']] for _ in src_infos]
                r = self._assign_objects(src_cands, gd)
                if r:
                    results.append((gd['purpose'], gk, r))
        return results

    def _assign_objects(self, src_candidate_cats, group_data):
        per_src_pool: List[List[Tuple[str, str]]] = []
        for cats in src_candidate_cats:
            pool: List[Tuple[str, str]] = []
            for cat in cats:
                for uid in self._find_assets_gather(cat):
                    pool.append((uid, cat))
            if not pool:
                return None
            per_src_pool.append(pool)

        used_uids: set = set()
        used_cats: set = set()
        assignments: List[Tuple[str, str, str]] = []

        for pool in per_src_pool:
            random.shuffle(pool)
            chosen = None
            for uid, cat in pool:
                if uid not in used_uids and cat not in used_cats:
                    chosen = (uid, cat); break
            if chosen is None:
                for uid, cat in pool:
                    if uid not in used_uids:
                        chosen = (uid, cat); break
            if chosen is None:
                return None
            uid, cat = chosen
            primary_cat = self.asset_library.get(uid, {}).get('category', cat)
            used_uids.add(uid); used_cats.add(cat)
            cap = self.asset_library.get(uid, {}).get('detailed_caption', '')
            if not cap:
                return None
            cap = _ensure_category_in_caption(cap, primary_cat)
            assignments.append((uid, primary_cat, cap))
        return assignments

    def _select_distractors(self, assignments, purpose_gk):
        used_uids = {a[0] for a in assignments}
        group_cats = {c.lower() for c in self.purpose_groups[purpose_gk].get('categories', [])}
        per_src_same: List[List[Tuple[str, str, str]]] = []
        per_src_diff: List[List[Tuple[str, str, str]]] = []

        for target_uid, target_cat, _ in assignments:
            same: List[Tuple[str, str, str]] = []
            pool = [u for u in self._find_assets_gather(target_cat) if u not in used_uids]
            random.shuffle(pool)
            for uid in pool:
                if len(same) >= self.num_same_cat_distractors:
                    break
                cap = self.asset_library.get(uid, {}).get('detailed_caption', '')
                if cap:
                    same.append((uid, target_cat, cap)); used_uids.add(uid)
            per_src_same.append(same)

            diff: List[Tuple[str, str, str]] = []
            diff_cats = [c for c in self.category_to_assets if c not in group_cats]
            random.shuffle(diff_cats)
            for cat in diff_cats:
                if len(diff) >= self.num_diff_cat_distractors:
                    break
                candidates = [u for u in self.category_to_assets[cat] if u not in used_uids]
                if not candidates:
                    continue
                uid = random.choice(candidates)
                cap = self.asset_library.get(uid, {}).get('detailed_caption', '')
                if cap:
                    diff.append((uid, cat, cap)); used_uids.add(uid)
            per_src_diff.append(diff)

        return per_src_same, per_src_diff

    def _build_world_graph(self, all_furniture, src_names, dest_name,
                           assignments, per_src_dist_uids, is_initial):
        wg = {}
        for fur in all_furniture:
            name = fur['name']
            entry = {"room": fur.get('room_name', ''), "content": []}
            if is_initial:
                for i, sn in enumerate(src_names):
                    if name == sn:
                        entry["content"] = [assignments[i][0]] + per_src_dist_uids[i]
            else:
                for i, sn in enumerate(src_names):
                    if name == sn:
                        entry["content"] = list(per_src_dist_uids[i])
                if name == dest_name:
                    entry["content"] = [a[0] for a in assignments]
            wg[name] = entry
        return wg

    def _build_execution_plan(self, src_names, dest_name, assignments, all_names):
        plan = []
        dest_distractors = self._get_furniture_distractors(dest_name, all_names)
        for i, sn in enumerate(src_names):
            uid, cat, cap = assignments[i]
            plan += self._nav_steps(self._get_furniture_distractors(sn, all_names), sn)
            plan += self._pick_steps(cat, cap)
            plan += self._nav_steps(dest_distractors, dest_name)
            plan.append(self._place_step(dest_name))
        return plan

    def _gen_specific_desc(self, assignments, src_refs, dest_caption):
        n = len(assignments)
        t = random.choice(GATHER_TEMPLATES_2 if n == 2 else GATHER_TEMPLATES_3)
        for i, (uid, cat, cap) in enumerate(assignments, 1):
            t = t.replace(f"[obj{i}_desc]", _fmt_obj_desc(cap))
        for i, sr in enumerate(src_refs, 1):
            t = t.replace(f"[src{i}]", sr.rstrip('.'))
        return t.replace("[dest]", dest_caption.rstrip('.'))

    def _gen_fuzzy_desc(self, purpose, src_refs, dest_caption):
        n = len(src_refs)
        t = random.choice(GATHER_FUZZY_TEMPLATES_2 if n == 2 else GATHER_FUZZY_TEMPLATES_3)
        t = t.replace("[purpose]", purpose)
        for i, sr in enumerate(src_refs, 1):
            t = t.replace(f"[src{i}]", sr.rstrip('.'))
        return t.replace("[dest]", dest_caption.rstrip('.'))

    def _build_single_task(self, scene_id, mode, src_furs, dest_fur, all_furniture,
                           generated_descs, room_counts, purpose_name, purpose_gk, assignments):
        per_src_same, per_src_diff = self._select_distractors(assignments, purpose_gk)
        per_src_dist_uids = [
            [u for u, _, _ in same] + [u for u, _, _ in diff]
            for same, diff in zip(per_src_same, per_src_diff)
        ]

        src_refs = [self._format_src_ref(sf, room_counts) for sf in src_furs]
        dest_caption = self._get_furniture_caption(dest_fur)
        src_names = [sf['name'] for sf in src_furs]
        dest_name = dest_fur['name']

        desc = self._gen_specific_desc(assignments, src_refs, dest_caption)
        if desc in generated_descs:
            self.stats['duplicate_skipped'] += 1
            return None
        generated_descs.add(desc)

        all_names = [f['name'] for f in all_furniture]
        obj_meta = {uid: {
            "category": cat, "detailed_caption": cap,
            "usd_path": self.asset_library.get(uid, {}).get('usd_path', ''),
            "size": self.asset_library.get(uid, {}).get('size', []),
        } for uid, cat, cap in assignments}

        dist_meta: Dict[str, Any] = {}
        per_src_dist_info: Dict[str, Dict] = {}
        for i, sn in enumerate(src_names):
            same_uids = [u for u, _, _ in per_src_same[i]]
            diff_uids = [u for u, _, _ in per_src_diff[i]]
            per_src_dist_info[sn] = {"same_category": same_uids, "diff_category": diff_uids}
            for items in (per_src_same[i], per_src_diff[i]):
                for uid, cat, cap in items:
                    a = self.asset_library.get(uid, {})
                    dist_meta[uid] = {
                        "category": cat, "detailed_caption": cap,
                        "usd_path": a.get('usd_path', ''), "size": a.get('size', []),
                    }

        task = {
            "scene_id": scene_id,
            "task_type": self.TASK_TYPE,
            "mode": mode,
            "task_description": desc,
            "fuzzy_task_description": self._gen_fuzzy_desc(purpose_name, src_refs, dest_caption),
            "purpose": purpose_name,
            "purpose_group": purpose_gk,
            "sources": [
                {"name": sf['name'], "room": sf.get('room_name', ''),
                 "ref": sr, "object_uid": assignments[i][0]}
                for i, (sf, sr) in enumerate(zip(src_furs, src_refs))
            ],
            "dest": dest_name,
            "dest_room": dest_fur.get('room_name', ''),
            "dest_caption": dest_caption,
            "initial_world_graph": self._build_world_graph(
                all_furniture, src_names, dest_name, assignments, per_src_dist_uids, True),
            "goal_world_graph": self._build_world_graph(
                all_furniture, src_names, dest_name, assignments, per_src_dist_uids, False),
            "src_furniture_distractors": {
                sn: self._get_furniture_distractors(sn, all_names) for sn in src_names},
            "dest_furniture_distractors": self._get_furniture_distractors(dest_name, all_names),
            "obj_meta": obj_meta,
            "obj_distractors": per_src_dist_info,
            "obj_distractor_meta": dist_meta,
            "execution_plan": self._build_execution_plan(src_names, dest_name, assignments, all_names),
        }
        passed, placements = self._check_layout(task)
        if not passed:
            self.stats['placement_check_failed'] += 1
            return None
        if placements:
            task['computed_placements'] = placements
        return task

    def _generate_tasks(self, scene_id, mode, src_furs, dest_fur, all_furniture,
                        generated_descs, room_counts):
        all_results = self._select_all_purpose_objects(src_furs, dest_fur)
        if not all_results:
            self.stats['skipped_no_purpose_objects'] += 1
            return []
        tasks = []
        for purpose_name, purpose_gk, assignments in all_results:
            task = self._build_single_task(
                scene_id, mode, src_furs, dest_fur, all_furniture,
                generated_descs, room_counts, purpose_name, purpose_gk, assignments)
            if task:
                tasks.append(task)
        return tasks

    def generate_tasks_for_scene(self, scene_id: str) -> List[Dict]:
        all_furniture = self._filter_valid_furniture(self.scene_furniture.get(scene_id, []))
        if len(all_furniture) < self.num_objects + 1:
            return []

        tasks = []
        generated_descs: set = set()
        room_counts = self._count_room_instances(scene_id)
        n_src = self.num_objects

        for _room, room_furniture in self._find_same_room_groups(scene_id):
            combos = list(itertools.combinations(room_furniture, n_src + 1))
            random.shuffle(combos)
            for combo in combos:
                combo = list(combo)
                for dest_idx in range(len(combo)):
                    dest_fur = combo[dest_idx]
                    src_furs = combo[:dest_idx] + combo[dest_idx + 1:]
                    for t in self._generate_tasks(scene_id, "same_room", src_furs, dest_fur,
                                                  all_furniture, generated_descs, room_counts):
                        tasks.append(t)
                        self.stats['tasks_same_room'] += 1

        for _ftype, typed_furniture in self._find_cross_room_groups(scene_id):
            by_room: Dict[str, List[Dict]] = defaultdict(list)
            for f in typed_furniture:
                by_room[f['room_name']].append(f)
            rooms = list(by_room.keys())
            for room_combo in itertools.combinations(rooms, n_src):
                src_options = [by_room[r] for r in room_combo]
                for src_pick in itertools.product(*src_options):
                    src_furs = list(src_pick)
                    src_names_set = {sf['name'] for sf in src_furs}
                    dest_candidates = [f for f in all_furniture
                                       if f['name'] not in src_names_set]
                    random.shuffle(dest_candidates)
                    for dest_fur in dest_candidates[:3]:
                        for t in self._generate_tasks(
                                scene_id, "cross_room", src_furs, dest_fur,
                                all_furniture, generated_descs, room_counts):
                            tasks.append(t)
                            self.stats['tasks_cross_room'] += 1

        self.stats['scenes_processed'] += 1
        return tasks


# ── Polish (optional LLM instruction rewriting) ───────────────────────────────

_POLISH_SYSTEM_PROMPT = (
    "You are a helpful assistant that rewrites robotic task instructions "
    "to sound natural and fluent in English. "
    "Keep the meaning and all location/object references identical. "
    "Do not add or remove information. Output only the rewritten sentence, no explanation."
)
_POLISH_BATCH_SIZE = 64
_POLISH_TIMEOUT = 60
_POLISH_MAX_RETRIES = 3


async def _polish_batch_async(texts: List[str], bar) -> List[str]:
    client = _AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
    )

    async def _one(text: str) -> str:
        for attempt in range(_POLISH_MAX_RETRIES):
            try:
                response = await _asyncio.wait_for(
                    client.chat.completions.create(
                        model="gemini-3-flash-preview",
                        messages=[
                            {"role": "system", "content": _POLISH_SYSTEM_PROMPT},
                            {"role": "user", "content": text},
                        ],
                        temperature=0.3,
                    ),
                    timeout=_POLISH_TIMEOUT,
                )
                bar.update(1)
                return response.choices[0].message.content.strip()
            except Exception:
                if attempt == _POLISH_MAX_RETRIES - 1:
                    bar.update(1)
                    return text
                await _asyncio.sleep(2 ** attempt)

    return await _asyncio.gather(*[_one(t) for t in texts])


def polish_tasks(tasks: List[Dict], fields: Optional[List[str]] = None) -> List[Dict]:
    """Polish task instruction fields in-place using an async LLM (requires openai package)."""
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("'openai' package is required for --polish")
    if fields is None:
        fields = ['task_description', 'fuzzy_task_description', 'detailed_task_description']

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    for field in fields:
        indices = [i for i, t in enumerate(tasks) if t.get(field)]
        if not indices:
            continue
        texts = [tasks[i][field] for i in indices]
        print(f"  Polishing {len(texts)} '{field}' entries ...")
        batches = [texts[j:j + _POLISH_BATCH_SIZE] for j in range(0, len(texts), _POLISH_BATCH_SIZE)]

        if _tqdm is not None:
            bar = _tqdm(total=len(texts), unit="item")
        else:
            class _Bar:
                def update(self, n=1): pass
                def close(self): pass
            bar = _Bar()

        async def _run_all():
            results = await _asyncio.gather(*[_polish_batch_async(b, bar) for b in batches])
            return [t for batch in results for t in batch]

        polished = _asyncio.run(_run_all())
        if _tqdm is not None:
            bar.close()
        for i, p in zip(indices, polished):
            tasks[i][field] = p
    return tasks


# ── YAML export (optional) ─────────────────────────────────────────────────────

class _LiteralStr(str):
    pass


class _FlowList(list):
    pass


def _represent_literal(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _represent_flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _make_yaml_dumper():
    class _CustomDumper(_yaml.Dumper):
        pass
    _CustomDumper.add_representer(_LiteralStr, _represent_literal)
    _CustomDumper.add_representer(_FlowList, _represent_flow_list)
    return _CustomDumper


_SCENE_USD_TMPL = "assets/scenes/{scene_id}_usd/scene.usd"
_OCC_MAP_TMPL = "assets/metadata/{scene_id}"
_FURNITURE_LIB_TMPL = "assets/metadata/{scene_id}/scene_furniture_library.json"


def _build_usd_scale_lookup(asset_library_path: str) -> Dict:
    lookup: Dict = {}
    p = Path(asset_library_path)
    if not p.exists():
        return lookup
    with open(p) as f:
        lib = json.load(f)
    for key, entry in lib.items():
        scale = entry.get("usd_scale")
        if not scale:
            continue
        oid_from_key = key.lstrip("_")
        lookup[oid_from_key] = scale
        oid_from_uid = (entry.get("original_uid") or "").replace("-", "_")
        if oid_from_uid and oid_from_uid not in lookup:
            lookup[oid_from_uid] = scale
    return lookup


def _convert_plan_to_str(plan_json: List[Dict], target_id: str) -> List[str]:
    out = []
    for step in plan_json:
        action = step["action"]
        args = step.get("args", {})
        if action == "nav_to":
            out.append(f"navigate to {args['receptacle_name']}")
        elif action == "find":
            desc = args.get("target_description")
            out.append(f"find {args['target_category']}" + (f" - {desc}" if desc else ""))
        elif action == "gaze_at":
            desc = args.get("target_description")
            out.append(f"gaze at {args['target_category']}" + (f" - {desc}" if desc else ""))
        elif action == "pick":
            desc = args.get("target_description")
            out.append(f"pick {args['target_category']}" + (f" - {desc}" if desc else "") + f" ({target_id})")
        elif action == "place":
            out.append(f"place on {args['receptacle_name']}")
        elif action == "open":
            out.append(f"open {args['receptacle_name']}")
        elif action == "close":
            out.append(f"close {args['receptacle_name']}")
        elif action == "ask":
            out.append(f"ask {args['target_category']} - {args['target_description']}")
        else:
            out.append(f"{action} {json.dumps(args)}")
    return out


def _strip_oid(oid: str) -> str:
    return oid.lstrip("_")


def _build_yaml_episode(task: Dict, task_idx: int, pos_to_key: Dict) -> Dict:
    task_type = task["task_type"]
    target_oids = [_strip_oid(k) for k in task["obj_meta"]]
    exec_plan = _convert_plan_to_str(task["execution_plan"], target_oids[0])

    placements: Dict = {}
    for placement_data in task.get("computed_placements", {}).values():
        oid = _strip_oid(placement_data["original_id"])
        pos = tuple(round(v, 6) for v in placement_data["position"])
        obj_key = pos_to_key.get((oid, pos))
        if obj_key:
            placements[obj_key] = {"original_id": oid, "furniture": placement_data["furniture"]}

    obj_distractor_meta = {
        _strip_oid(k): {"category": v["category"]}
        for k, v in task.get("obj_distractor_meta", {}).items()
    }

    episode: Dict = {
        "task_id": f"task_{task_idx}",
        "task_type": task_type,
        "task_description": task["task_description"],
    }

    if task_type == "gather":
        episode["fuzzy_task_description"] = task.get("fuzzy_task_description", "")
        episode["mode"] = task.get("mode", "")
        episode["purpose"] = task.get("purpose", "")
        episode["purpose_group"] = task.get("purpose_group", "")
        episode["sources"] = task["sources"]
        episode["dest"] = task["dest"]
        episode["dest_room"] = task.get("dest_room", "")
        episode["dest_caption"] = task.get("dest_caption", "")
        episode["target_object_ids"] = target_oids
        episode["src_furniture_distractors"] = task.get("src_furniture_distractors", {})
        episode["dest_furniture_distractors"] = task.get("dest_furniture_distractors", [])
    elif task_type == "interactive":
        episode["detailed_task_description"] = task.get("detailed_task_description", "")
        episode["purpose"] = task.get("purpose", "")
        episode["purpose_group"] = task.get("purpose_group", "")
        episode["src"] = task["src"]
        episode["dest"] = task["dest"]
        episode["target_object_id"] = target_oids[0]
        episode["src_distractors"] = task.get("src_distractors", [])
        episode["dest_distractors"] = task.get("dest_distractors", [])
    else:
        episode["src"] = task["src"]
        episode["dest"] = task["dest"]
        episode["target_object_id"] = target_oids[0]
        episode["src_distractors"] = task.get("src_distractors", [])
        episode["dest_distractors"] = task.get("dest_distractors", [])

    episode["obj_distractors"] = [_strip_oid(oid) for oid in task.get("obj_distractors", [])]
    episode["obj_distractor_meta"] = obj_distractor_meta
    episode["execution_plan"] = exec_plan
    episode["placements"] = placements
    return episode


def export_tasks_to_yaml(all_tasks: List[Dict], out_dir: str, asset_library_path: str):
    """Export tasks to per-scene per-type YAML files (SFT format)."""
    if not _YAML_AVAILABLE:
        raise RuntimeError("'yaml' (PyYAML) package is required for --to-yaml")

    usd_scale_lookup = _build_usd_scale_lookup(asset_library_path)
    Dumper = _make_yaml_dumper()

    # Group by (scene_id, task_type)
    groups: Dict[Tuple, List] = defaultdict(list)
    for task in all_tasks:
        groups[(task["scene_id"], task["task_type"])].append(task)

    total_episodes = 0
    for (scene_id, task_type), task_list in sorted(groups.items()):
        scene_dir = Path(out_dir) / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"{task_type}.yaml"

        all_object_meta: Dict = {}
        for task in task_list:
            combined_meta = {**task["obj_meta"], **task.get("obj_distractor_meta", {})}
            for oid_key, meta in combined_meta.items():
                oid = _strip_oid(oid_key)
                if oid not in all_object_meta:
                    all_object_meta[oid] = {
                        "category": meta["category"],
                        "usd_path": meta["usd_path"],
                        "size": [round(v, 6) for v in meta["size"]],
                    }

        all_placement_instances: Dict = {}
        for task in task_list:
            for placement_data in task.get("computed_placements", {}).values():
                oid = _strip_oid(placement_data["original_id"])
                pos = tuple(round(v, 6) for v in placement_data["position"])
                all_placement_instances[(oid, pos)] = True

        objects: Dict = {}
        pos_to_key: Dict = {}
        for idx, (oid, pos) in enumerate(sorted(all_placement_instances.keys())):
            meta = all_object_meta.get(oid)
            if meta is None:
                continue
            obj_key = f"obj_{idx}"
            entry: Dict = {
                "original_id": oid,
                "category": meta["category"],
                "usd_path": meta["usd_path"],
            }
            if oid in usd_scale_lookup:
                entry["usd_scale"] = _FlowList(usd_scale_lookup[oid])
            entry["position"] = list(pos)
            entry["size"] = meta["size"]
            objects[obj_key] = entry
            pos_to_key[(oid, pos)] = obj_key

        episodes = [_build_yaml_episode(task, idx, pos_to_key) for idx, task in enumerate(task_list)]

        doc = {
            "scene_id": scene_id,
            "task_type": task_type,
            "paths": {
                "scene_usd": _SCENE_USD_TMPL.format(scene_id=scene_id),
                "occ_map_dir": _OCC_MAP_TMPL.format(scene_id=scene_id),
                "furniture_lib": _FURNITURE_LIB_TMPL.format(scene_id=scene_id),
            },
            "objects": objects,
            "episodes": episodes,
        }

        with open(out_path, "w") as f:
            _yaml.dump(doc, f, Dumper=Dumper, default_flow_style=False,
                       allow_unicode=True, sort_keys=False, width=120)

        total_episodes += len(episodes)
        print(f"[yaml]  {out_path}  ({len(episodes)} episodes, {len(objects)} objects)")

    print(f"YAML export complete: {total_episodes} total episodes → {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

TASK_TYPES = ['basic', 'distractor', 'articulation', 'interactive', 'gather']


def _print_stats(task_type: str, gen: BasePnPGenerator, n_tasks: int):
    print(f"\n{'=' * 55}")
    print(f"{task_type.upper()} — {n_tasks} tasks generated")
    print(f"{'=' * 55}")
    for k, v in sorted(gen.stats.items()):
        print(f"  {k:<40} {v}")


def main():
    parser = argparse.ArgumentParser(
        description='All-in-one PnP task generator (basic/distractor/articulation/interactive/gather)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data paths
    parser.add_argument('--furniture-library', default='assets/metadata',
                        help='Path to scene_furniture_library.json or assets/metadata root dir')
    parser.add_argument('--asset-library',
                        default='assets/metadata/consolidated_asset_library_with_size.json')
    parser.add_argument('--object-mapping', default='assets/metadata/furniture_pair_object_mapping.json')
    parser.add_argument('--category-pairs', default='assets/metadata/category_pairs_same_purpose.json',
                        help='Required for interactive and gather')

    # Task selection
    parser.add_argument('--tasks', nargs='+',
                        choices=TASK_TYPES + ['all'], default=['all'],
                        help='Task types to generate')

    # Output
    parser.add_argument('--output-dir', '-o', default='proc_datagen/configs',
                        help='Directory for output YAML files '
                             '(per-scene per-type: {output_dir}/{scene_id}/{task_type}.yaml)')

    # Per-type options
    parser.add_argument('--num-distractors', type=int, default=2,
                        help='Object distractors per task (distractor/interactive)')
    parser.add_argument('--max-objects-per-pair', type=int, default=5,
                        help='Max objects per furniture pair (articulation)')
    parser.add_argument('--num-objects', type=int, default=2, choices=[2, 3],
                        help='Objects per gather task')
    parser.add_argument('--same-distractors', type=int, default=1,
                        help='Same-category distractors per source (gather)')
    parser.add_argument('--diff-distractors', type=int, default=1,
                        help='Diff-category distractors per source (gather)')
    parser.add_argument('--max-purposes', type=int, default=3,
                        help='Max purpose groups per furniture combo (gather)')

    # Scene selection
    parser.add_argument('--scene-ids', nargs='+',
                        help='Specific scene IDs (default: all)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of scenes to process')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats without saving')

    # Placement verification
    parser.add_argument('--verify-placement', action='store_true',
                        help='Filter tasks where objects cannot physically fit on furniture '
                             '(checks src with target+distractors, and each src_distractor '
                             'with target alone)')
    parser.add_argument('--occ-map-root', default=None,
                        help='Root dir of per-scene occupancy maps '
                             '(<root>/<scene_id>/occupancy.npy). '
                             'If omitted, placement is checked using bbox only.')

    # Polish
    parser.add_argument('--polish', action='store_true',
                        help='Polish task descriptions with an LLM after generation '
                             '(requires openai package and OPENAI_API_KEY env var)')
    parser.add_argument('--polish-fields', nargs='+',
                        default=['task_description', 'fuzzy_task_description',
                                 'detailed_task_description'],
                        help='Task fields to polish (default: task_description '
                             'fuzzy_task_description detailed_task_description)')

    # JSON export (backward compat)
    parser.add_argument('--to-json', action='store_true',
                        help='Also save flat JSON files per task type '
                             '(for backward compatibility)')

    args = parser.parse_args()

    task_types = TASK_TYPES if 'all' in args.tasks else list(dict.fromkeys(args.tasks))

    common = dict(
        furniture_library_path=args.furniture_library,
        asset_library_path=args.asset_library,
        object_mapping_path=args.object_mapping,
        seed=args.seed,
        verify_placement=args.verify_placement,
        occ_map_root=args.occ_map_root,
    )

    generators: Dict[str, BasePnPGenerator] = {}
    init_map = {
        'basic':        lambda: BasicPnPGenerator(**common),
        'distractor':   lambda: DistractorPnPGenerator(
            num_distractors=args.num_distractors, **common),
        'articulation': lambda: ArticulationPnPGenerator(
            max_objects_per_pair=args.max_objects_per_pair, **common),
        'interactive':  lambda: InteractivePnPGenerator(
            category_pairs_path=args.category_pairs,
            num_distractors=args.num_distractors, **common),
        'gather':       lambda: GatherPnPGenerator(
            category_pairs_path=args.category_pairs,
            num_objects=args.num_objects,
            num_same_cat_distractors=args.same_distractors,
            num_diff_cat_distractors=args.diff_distractors,
            max_purposes_per_combo=args.max_purposes, **common),
    }

    for tt in task_types:
        print(f"Initializing {tt} generator...")
        generators[tt] = init_map[tt]()

    # Resolve scene IDs using the first generator's scene list
    first_gen = next(iter(generators.values()))
    scene_ids = args.scene_ids
    if scene_ids is None and args.limit:
        scene_ids = list(first_gen.scene_furniture.keys())[:args.limit]

    output_dir = Path(args.output_dir)
    total_tasks = 0
    all_generated: Dict[str, List[Dict]] = {}

    for tt, gen in generators.items():
        print(f"\nGenerating {tt} tasks...")
        tasks = gen.generate_tasks(scene_ids)
        _print_stats(tt, gen, len(tasks))
        total_tasks += len(tasks)
        all_generated[tt] = tasks

        if args.dry_run:
            print(f"  [dry-run] {len(tasks)} tasks not saved")
            if tasks:
                sample = json.dumps(tasks[0], indent=2, ensure_ascii=False)
                print(f"  Sample:\n{sample[:600]}{'...' if len(sample) > 600 else ''}")

    print(f"\nDone. Total tasks across all types: {total_tasks}")

    # ── Polish ────────────────────────────────────────────────────────────────
    if args.polish and not args.dry_run:
        print("\n── Polishing task descriptions ──")
        for tt, tasks in all_generated.items():
            if not tasks:
                continue
            print(f"\n[{tt}]")
            all_generated[tt] = polish_tasks(tasks, args.polish_fields)

    # ── YAML export (default) ────────────────────────────────────────────────
    if not args.dry_run:
        print(f"\n── Exporting to YAML → {output_dir} ──")
        flat_tasks = [t for tasks in all_generated.values() for t in tasks]
        export_tasks_to_yaml(flat_tasks, str(output_dir), args.asset_library)
        print(f"Output directory: {output_dir.resolve()}")

    # ── JSON export (optional, backward compat) ──────────────────────────────
    if args.to_json and not args.dry_run:
        json_dir = output_dir / "json"
        print(f"\n── Also saving JSON → {json_dir} ──")
        for tt, gen in generators.items():
            gen.save_tasks(all_generated[tt], str(json_dir / f"{tt}_tasks.json"))


if __name__ == '__main__':
    main()
