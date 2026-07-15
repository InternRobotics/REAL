#!/usr/bin/env python3
"""Build a portable release of REAL's seven processed GRScenes scenes.

InternUtopia's ``export_scenes.py`` is deliberately used for the initial
model/material collection.  That upstream helper copies authored USD files
verbatim, however, so a second strict pass is required to select REAL's
processed stage, localize every USD asset path, remove export-time symlinks,
and prove that the resulting dependency closure is self-contained.

This script must run in a Python environment that can import ``pxr``.  It does
not start Isaac Sim or require a display.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCENE_IDS = (
    "MV7J6NIKTKJZ2AABAAAAAEI8",
    "MVUCSQAKTKJ5EAABAAAAAAI8",
    "MVUCSQAKTKJ5EAABAAAAAAQ8",
    "MVUCSQAKTKJ5EAABAAAAABA8",
    "MVUCSQAKTKJ5EAABAAAAABQ8",
    "MVUCSQAKTKJ5EAABAAAAABY8",
    "MVUCSQAKTKJ5EAABAAAAACA8",
)

ABY8_SCENE_ID = "MVUCSQAKTKJ5EAABAAAAABY8"
EXPECTED_PROCESSED_SHA256 = {
    "MV7J6NIKTKJZ2AABAAAAAEI8": "97f1075ab47435fce5caa85d8ca39e8718ea67eaedb2d2fa153cf03b16388446",
    "MVUCSQAKTKJ5EAABAAAAAAI8": "dedc0c9275561021ed33b8e85aff7ddb18ee1d90bf3b6a3c0b74c1276e016ad4",
    "MVUCSQAKTKJ5EAABAAAAAAQ8": "624002a8add4cced60da4c3f671c0446e4d34ff61159797616f2cd9057a99dc1",
    "MVUCSQAKTKJ5EAABAAAAABA8": "004eb524e75300253d11bb230ecc772498b6bbccd51acc470a6a1f65caee256b",
    "MVUCSQAKTKJ5EAABAAAAABQ8": "2c82a6c8fcb347b68793440abf6fc8f0f5f9bfa360bbded7ea792836def383f1",
    "MVUCSQAKTKJ5EAABAAAAABY8": "bf0647d0bba61dbb4359b8a9ab42468310beb754588b1e9e7fa64785e3289bf6",
    "MVUCSQAKTKJ5EAABAAAAACA8": "85ec219aa2f0a130b197ba083d067ba912e8f2d0093119056523dea0901860a2",
}
EXPECTED_SOURCE_SHA256 = {
    **EXPECTED_PROCESSED_SHA256,
    ABY8_SCENE_ID: "02ab742fef9f0669278b560c03a54f63e3a82db16dbd249834709a3fbe776a43",
}
EXPECTED_RAW_SEED_SHA256 = {
    "MV7J6NIKTKJZ2AABAAAAAEI8": "31beb77ff3ebeb75d16b053e2c0407a4aeebd6582a53117365c0bfbfff49f12a",
    "MVUCSQAKTKJ5EAABAAAAAAI8": "07a2684ee0cd3479c07093e506fdb88d8aa0a75185936042253eadf4d3020177",
    "MVUCSQAKTKJ5EAABAAAAAAQ8": "c68e6aaa891c0e75e75a89c0b82e74cf98db20a6854026e1e52066feed6c523e",
    "MVUCSQAKTKJ5EAABAAAAABA8": "1b79b081ffb94f6fe76979018d86ca8828c04dc5f769a3b3fd6dd5da91841418",
    "MVUCSQAKTKJ5EAABAAAAABQ8": "8c497c452b82531959fde035a4789c5aa78b9951089b0e0d404e38bbf9ec622a",
    "MVUCSQAKTKJ5EAABAAAAABY8": "986cc448f626255c068c2875fc350a21ac9e56a0dc4e65e22bece9ddf2143d12",
    "MVUCSQAKTKJ5EAABAAAAACA8": "4233a0eb3ba2f367c01b2190982f514895ae772d8b49b30056fc2f1a7547e3be",
}
EXPECTED_TOOLKIT_EXPORTER_SHA256 = (
    "cd0db9b5206901f56e68bbeda3b1a65785363fc6db2da9235e63b769e8f987f4"
)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
PATH_MARKERS = ("models", "Materials", "scenes", "metadata")
GLOBAL_METADATA_FILES = (
    "category_pairs_same_purpose.json",
    "consolidated_asset_library_with_size.json",
    "furniture_pair_object_mapping.json",
)
SCENE_METADATA_FILES = ("occupancy.npy", "scene_furniture_library.json")
MESA_REQUIRED_OBJECTS = (
    "c13555900d7f413bad3caec2086d3874.usd",
    "052fedd7-cb75-43dd-9685-bd85a0e1619b.usd",
)
PRIVATE_PATH_MARKERS = (b"/cpfs/", b"/shared/")
PATH_BEARING_SUFFIXES = frozenset(
    {".usd", ".usda", ".usdc", ".mdl", ".json", ".yaml", ".yml", ".md", ".txt"}
)
_SCENE_ID_RE = re.compile(r"^[A-Za-z0-9]+$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MDL_RESOURCE_RE = re.compile(
    r"(?:texture_(?:2d|3d|cube|ptex)|light_profile|bsdf_measurement)"
    r'\(\s*"([^"]+)"'
)
_MDL_LOCAL_MODULE_RE = re.compile(r"(?m)^\s*(?:import|using)\s+(\.?)::([A-Za-z_][A-Za-z0-9_]*)")
_MDL_BUILTIN_MODULES = frozenset({"anno", "base", "df", "math", "state", "tex"})


class ReleaseError(RuntimeError):
    """Raised when a release safety or validation gate fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def validate_scene_ids(scene_ids: Sequence[str]) -> tuple[str, ...]:
    if not scene_ids:
        raise ReleaseError("at least one scene id is required")
    if len(scene_ids) != len(set(scene_ids)):
        raise ReleaseError("scene ids must be unique")
    invalid = [scene_id for scene_id in scene_ids if not _SCENE_ID_RE.fullmatch(scene_id)]
    if invalid:
        raise ReleaseError(f"unsafe scene ids: {invalid}")
    unsupported = [scene_id for scene_id in scene_ids if scene_id not in SCENE_IDS]
    if unsupported:
        raise ReleaseError(f"unsupported REAL processed scene ids: {unsupported}")
    return tuple(scene_ids)


def require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ReleaseError(f"{label} is not a directory: {resolved}")
    return resolved


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ReleaseError(f"{label} is not a file: {resolved}")
    return resolved


def get_git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip() or None

    result = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    top_level = Path(result.stdout.strip())
    commit = subprocess.run(
        ["git", "-C", str(top_level), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return commit.stdout.strip() or None if commit.returncode == 0 else None


def validate_inputs(
    source_root: Path,
    processed_scenes_root: Path,
    metadata_root: Path,
    toolkit_exporter: Path,
    output: Path,
    scene_ids: Sequence[str],
    source_stage_name: str,
    processed_stage_name: str,
) -> dict[str, dict[str, Any]]:
    if output.exists() or output.is_symlink():
        raise ReleaseError(f"refusing to overwrite existing output: {output}")
    for protected in (source_root, processed_scenes_root, metadata_root):
        if is_within(output, protected):
            raise ReleaseError(f"output must not be inside input tree: {protected}")

    provenance: dict[str, dict[str, Any]] = {}
    for scene_id in scene_ids:
        raw_seed_stage = source_root / "scenes" / f"{scene_id}_usd" / "start_result_raw.usd"
        source_stage = source_root / "scenes" / f"{scene_id}_usd" / source_stage_name
        processed_stage = processed_scenes_root / f"{scene_id}_usd" / processed_stage_name
        require_file(raw_seed_stage, f"InternUtopia raw seed stage for {scene_id}")
        require_file(source_stage, f"processed source stage for {scene_id}")
        require_file(processed_stage, f"processed stage for {scene_id}")
        for filename in SCENE_METADATA_FILES:
            require_file(
                metadata_root / scene_id / filename,
                f"{filename} metadata for {scene_id}",
            )
        raw_seed_hash = sha256_file(raw_seed_stage)
        source_hash = sha256_file(source_stage)
        processed_hash = sha256_file(processed_stage)
        expected_raw_seed = EXPECTED_RAW_SEED_SHA256[scene_id]
        expected_source = EXPECTED_SOURCE_SHA256[scene_id]
        expected_processed = EXPECTED_PROCESSED_SHA256[scene_id]
        if raw_seed_hash != expected_raw_seed:
            raise ReleaseError(
                f"raw seed stage hash mismatch for {scene_id}: "
                f"expected {expected_raw_seed}, got {raw_seed_hash}"
            )
        if source_hash != expected_source:
            raise ReleaseError(
                f"source stage hash mismatch for {scene_id}: "
                f"expected {expected_source}, got {source_hash}"
            )
        if processed_hash != expected_processed:
            raise ReleaseError(
                f"processed stage hash mismatch for {scene_id}: "
                f"expected {expected_processed}, got {processed_hash}"
            )
        provenance[scene_id] = {
            "internutopia_seed_stage": "start_result_raw.usd",
            "internutopia_seed_sha256": raw_seed_hash,
            "source_stage": source_stage_name,
            "source_sha256": source_hash,
            "processed_entry": processed_stage_name,
            "processed_sha256": processed_hash,
        }

    for filename in GLOBAL_METADATA_FILES:
        require_file(metadata_root / filename, f"global metadata {filename}")
    require_file(toolkit_exporter, "InternUtopia exporter")
    exporter_hash = sha256_file(toolkit_exporter)
    if exporter_hash != EXPECTED_TOOLKIT_EXPORTER_SHA256:
        raise ReleaseError(
            "InternUtopia exporter hash mismatch: "
            f"expected {EXPECTED_TOOLKIT_EXPORTER_SHA256}, got {exporter_hash}"
        )
    return provenance


def run_internutopia_exporter(
    toolkit_exporter: Path,
    source_root: Path,
    target_root: Path,
    scene_ids: Sequence[str],
) -> None:
    command = [
        sys.executable,
        str(toolkit_exporter),
        "--input",
        str(source_root),
        "--output",
        str(target_root),
        "--names",
        *(f"{scene_id}_usd" for scene_id in scene_ids),
    ]
    print("[1/7] Running InternUtopia export_scenes.py", flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise ReleaseError(f"InternUtopia exporter failed with exit code {result.returncode}")
    # Upstream copy_file() swallows errors and normally still exits zero.
    if "Error!" in combined:
        raise ReleaseError("InternUtopia exporter reported a swallowed copy error")


def select_processed_stages(
    target_root: Path,
    processed_scenes_root: Path,
    scene_ids: Sequence[str],
    processed_stage_name: str,
) -> None:
    print("[2/7] Selecting experiment-processed scene entry points", flush=True)
    for scene_id in scene_ids:
        scene_dir = target_root / "scenes" / f"{scene_id}_usd"
        if not scene_dir.is_dir():
            raise ReleaseError(f"toolkit did not create scene directory: {scene_dir}")
        for path in scene_dir.iterdir():
            if path.is_file() and path.suffix.lower() in USD_SUFFIXES:
                path.unlink()
        source = processed_scenes_root / f"{scene_id}_usd" / processed_stage_name
        shutil.copy2(source, scene_dir / "scene.usd")


def copy_runtime_metadata(target_root: Path, metadata_root: Path, scene_ids: Sequence[str]) -> None:
    target_metadata = target_root / "metadata"
    target_metadata.mkdir(parents=True, exist_ok=True)
    for filename in GLOBAL_METADATA_FILES:
        shutil.copy2(metadata_root / filename, target_metadata / filename)
    for scene_id in scene_ids:
        scene_target = target_metadata / scene_id
        scene_target.mkdir(parents=True, exist_ok=True)
        for filename in SCENE_METADATA_FILES:
            shutil.copy2(metadata_root / scene_id / filename, scene_target / filename)


class AssetLocator:
    """Resolve authored asset tokens to regular files inside an export tree."""

    def __init__(self, target_root: Path):
        self.target_root = target_root.resolve(strict=True)
        self.files_by_name: dict[str, list[Path]] = defaultdict(list)
        self.refresh_index()

    def refresh_index(self) -> None:
        self.files_by_name.clear()
        for path in self.target_root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                self.files_by_name[path.name].append(path.resolve())

    def _accept(self, candidate: Path) -> Path | None:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file() and is_within(resolved, self.target_root):
            return resolved
        return None

    def resolve(self, layer_path: Path, asset_path: str) -> tuple[Path | None, str]:
        if not asset_path:
            return None, "empty"
        if "://" in asset_path or asset_path.startswith("anon:"):
            return None, "unsupported-uri"

        token = Path(asset_path)
        direct = self._accept(layer_path.parent / token)
        if direct is not None:
            return direct, "direct"

        parts = token.parts
        for marker in PATH_MARKERS:
            indices = [index for index, part in enumerate(parts) if part == marker]
            for index in reversed(indices):
                marked = self._accept(self.target_root / Path(*parts[index:]))
                if marked is not None:
                    return marked, f"marker:{marker}"

        # Processed scene roots contain one known legacy path family that
        # points at a non-existent per-model Textures directory.  InternUtopia
        # ships the shared replacement in Materials/Textures.
        if token.name.startswith("T_Default_Material_Grid_"):
            shared_texture = self._accept(self.target_root / "Materials" / "Textures" / token.name)
            if shared_texture is not None:
                return shared_texture, "shared-default-texture"
        return None, "not-found"


def _load_pxr():
    try:
        from pxr import Sdf, Usd, UsdUtils
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise ReleaseError(
            "OpenUSD Python bindings are unavailable; activate the project "
            "environment and make pxr importable"
        ) from exc
    return Sdf, Usd, UsdUtils


def usd_layer_paths(target_root: Path) -> list[Path]:
    return sorted(
        path
        for path in target_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in USD_SUFFIXES
    )


def localize_usd_asset_paths(target_root: Path) -> dict[str, int]:
    _, _, UsdUtils = _load_pxr()
    from pxr import Sdf

    locator = AssetLocator(target_root)
    counters: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []
    layers = usd_layer_paths(target_root)
    print(f"[3/7] Localizing asset paths in {len(layers)} USD layers", flush=True)

    for index, layer_path in enumerate(layers, start=1):
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            raise ReleaseError(f"failed to open USD layer: {layer_path}")

        cache: dict[str, str] = {}

        def rewrite(asset_path: str) -> str:
            if asset_path in cache:
                counters["cache_hits"] += 1
                return cache[asset_path]
            resolved, method = locator.resolve(layer_path, asset_path)
            counters[method] += 1
            if resolved is None:
                failures.append((str(layer_path.relative_to(target_root)), asset_path, method))
                cache[asset_path] = asset_path
                return asset_path
            relative = os.path.relpath(resolved, layer_path.parent).replace(os.sep, "/")
            cache[asset_path] = relative
            if relative != asset_path:
                counters["rewritten"] += 1
            return relative

        UsdUtils.ModifyAssetPaths(layer, rewrite)
        # Saving a crate layer in place can retain dead entries in its string
        # table, including the old private absolute paths.  Export to a fresh
        # crate and atomically replace the copy so byte-level scans are clean.
        temporary_layer = layer_path.with_name(
            f".{layer_path.stem}.localized-{os.getpid()}{layer_path.suffix}"
        )
        if not layer.Export(str(temporary_layer)):
            raise ReleaseError(f"failed to export localized USD layer: {layer_path}")
        os.replace(temporary_layer, layer_path)
        if index % 500 == 0:
            print(f"  localized {index}/{len(layers)} layers", flush=True)

    if failures:
        examples = "\n".join(
            f"  {layer}: {asset} ({reason})" for layer, asset, reason in failures[:20]
        )
        raise ReleaseError(f"could not localize {len(failures)} authored asset paths:\n{examples}")

    # InternUtopia creates convenience links.  All paths now point directly to
    # regular files, so remove links to make archive extraction deterministic.
    for path in sorted(target_root.rglob("*"), reverse=True):
        if path.is_symlink():
            path.unlink()
    return dict(sorted(counters.items()))


def collect_mdl_dependencies(initial_paths: Iterable[Path], target_root: Path) -> set[Path]:
    """Collect texture resources and local MDL modules recursively."""

    root = target_root.resolve(strict=True)
    dependencies: set[Path] = set()
    pending = [path.resolve() for path in initial_paths if path.suffix.lower() == ".mdl"]
    visited: set[Path] = set()
    errors: list[str] = []

    while pending:
        mdl_path = pending.pop()
        if mdl_path in visited:
            continue
        visited.add(mdl_path)
        if not mdl_path.is_file() or not is_within(mdl_path, root):
            errors.append(f"missing or external MDL module: {mdl_path}")
            continue
        text = mdl_path.read_text(encoding="utf-8-sig", errors="strict")

        for asset_path in _MDL_RESOURCE_RE.findall(text):
            if _path_is_absolute(asset_path) or "://" in asset_path:
                errors.append(f"non-relative MDL texture: {mdl_path} -> {asset_path}")
                continue
            resolved = (mdl_path.parent / asset_path).resolve(strict=False)
            if not resolved.is_file() or not is_within(resolved, root):
                errors.append(f"missing/external MDL texture: {mdl_path} -> {asset_path}")
                continue
            dependencies.add(resolved)

        for explicit_local, module_name in _MDL_LOCAL_MODULE_RE.findall(text):
            module_path = (mdl_path.parent / f"{module_name}.mdl").resolve(strict=False)
            if module_path.is_file() and is_within(module_path, root):
                if module_path not in dependencies:
                    dependencies.add(module_path)
                    pending.append(module_path)
            elif explicit_local or module_name not in _MDL_BUILTIN_MODULES:
                errors.append(f"missing/external local MDL module: {mdl_path} -> {module_name}")

    if errors:
        raise ReleaseError("MDL dependency validation failed:\n  " + "\n  ".join(errors[:50]))
    return dependencies


def prune_to_scene_closure(target_root: Path, scene_ids: Sequence[str]) -> dict[str, int]:
    """Delete the raw-stage dependency superset not used by processed entries."""

    Sdf, _, UsdUtils = _load_pxr()
    root = target_root.resolve(strict=True)
    keep: set[Path] = set()
    unresolved_errors: list[str] = []

    print("[4/7] Pruning the raw-stage seed to the processed-scene closure", flush=True)
    for scene_id in scene_ids:
        entry = root / "scenes" / f"{scene_id}_usd" / "scene.usd"
        keep.add(entry.resolve())
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(entry)))
        if unresolved:
            unresolved_errors.append(f"{scene_id}: {sorted(set(map(str, unresolved)))[:10]}")
        for layer in layers:
            layer_path = Path(_layer_real_path(layer)).resolve(strict=False)
            if not layer_path.is_file() or not is_within(layer_path, root):
                unresolved_errors.append(f"{scene_id}: external layer {layer_path}")
            else:
                keep.add(layer_path)
        for asset in assets:
            asset_path = Path(str(asset)).resolve(strict=False)
            if not asset_path.is_file() or not is_within(asset_path, root):
                unresolved_errors.append(f"{scene_id}: external asset {asset_path}")
            else:
                keep.add(asset_path)

    if unresolved_errors:
        raise ReleaseError(
            "cannot prune an incomplete processed-scene closure:\n  "
            + "\n  ".join(unresolved_errors[:50])
        )

    keep.update(collect_mdl_dependencies(keep, root))
    removable_roots = (root / "models", root / "Materials")
    removed_files = 0
    removed_bytes = 0
    kept_files = 0
    kept_bytes = 0
    for removable_root in removable_roots:
        for path in removable_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            size = path.stat().st_size
            if resolved in keep:
                kept_files += 1
                kept_bytes += size
            else:
                path.unlink()
                removed_files += 1
                removed_bytes += size
        for directory in sorted(
            (path for path in removable_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

    return {
        "kept_dependency_files": kept_files,
        "kept_dependency_bytes": kept_bytes,
        "removed_seed_files": removed_files,
        "removed_seed_bytes": removed_bytes,
        "mdl_dependency_files": len(collect_mdl_dependencies(keep, root)),
    }


def _path_is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or bool(_WINDOWS_ABSOLUTE_RE.match(value))


def _layer_real_path(layer) -> str:
    for attribute in ("realPath", "identifier"):
        value = getattr(layer, attribute, "")
        if value:
            return str(value)
    return ""


def stage_structure(path: Path) -> dict[str, int | str]:
    """Return a stable fingerprint of composed prim paths and type names."""

    _, Usd, _ = _load_pxr()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ReleaseError(f"stage failed to open: {path}")
    digest = hashlib.sha256()
    prim_count = 0
    for prim in stage.Traverse():
        record = f"{prim.GetPath()}\t{prim.GetTypeName()}\n".encode("utf-8")
        digest.update(record)
        prim_count += 1
    return {"composed_prims": prim_count, "path_type_sha256": digest.hexdigest()}


def _contains_private_marker(path: Path) -> bool:
    overlap = max(map(len, PRIVATE_PATH_MARKERS)) - 1
    tail = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            data = tail + chunk
            if any(marker in data for marker in PRIVATE_PATH_MARKERS):
                return True
            tail = data[-overlap:]
    return False


def verify_release(
    target_root: Path,
    scene_ids: Sequence[str],
    expected_structures: dict[str, dict[str, int | str]] | None = None,
) -> dict:
    Sdf, _, UsdUtils = _load_pxr()
    root = target_root.resolve(strict=True)
    errors: list[str] = []
    authored_asset_count = 0

    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        errors.append(f"release contains {len(symlinks)} symlinks")

    for layer_path in usd_layer_paths(root):
        layer = Sdf.Layer.FindOrOpen(str(layer_path))
        if layer is None:
            errors.append(f"cannot reopen USD layer {layer_path.relative_to(root)}")
            continue

        def inspect(asset_path: str) -> str:
            nonlocal authored_asset_count
            authored_asset_count += 1
            if not asset_path:
                return asset_path
            label = f"{layer_path.relative_to(root)} -> {asset_path}"
            if _path_is_absolute(asset_path) or "://" in asset_path:
                errors.append(f"non-relative authored asset path: {label}")
                return asset_path
            resolved = (layer_path.parent / asset_path).resolve(strict=False)
            if not resolved.is_file():
                errors.append(f"missing authored asset: {label}")
            elif not is_within(resolved, root):
                errors.append(f"out-of-root authored asset: {label}")
            return asset_path

        UsdUtils.ModifyAssetPaths(layer, inspect)

    private_files = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in PATH_BEARING_SUFFIXES
        and _contains_private_marker(path)
    ]
    if private_files:
        errors.append(f"private path markers remain in {private_files[:20]}")

    mdl_paths = list(root.rglob("*.mdl"))
    try:
        mdl_dependencies = collect_mdl_dependencies(mdl_paths, root)
    except ReleaseError as exc:
        errors.append(str(exc))
        mdl_dependencies = set()

    scene_reports: dict[str, dict[str, int]] = {}
    print("[5/7] Verifying composed dependency closure", flush=True)
    for scene_id in scene_ids:
        entry = root / "scenes" / f"{scene_id}_usd" / "scene.usd"
        if not entry.is_file():
            errors.append(f"missing entry point: {entry.relative_to(root)}")
            continue
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(entry)))
        unresolved_paths = sorted(set(map(str, unresolved)))
        if unresolved_paths:
            errors.append(
                f"{scene_id}: {len(unresolved_paths)} unresolved dependencies; "
                f"examples={unresolved_paths[:10]}"
            )

        external_layers = [
            _layer_real_path(layer)
            for layer in layers
            if not is_within(Path(_layer_real_path(layer)), root)
        ]
        external_assets = [str(asset) for asset in assets if not is_within(Path(str(asset)), root)]
        if external_layers:
            errors.append(f"{scene_id}: external layers {external_layers[:10]}")
        if external_assets:
            errors.append(f"{scene_id}: external assets {external_assets[:10]}")

        try:
            released_structure = stage_structure(entry)
        except ReleaseError as exc:
            errors.append(f"{scene_id}: {exc}")
            released_structure = {"composed_prims": 0, "path_type_sha256": ""}
        expected = (expected_structures or {}).get(scene_id)
        if expected is not None and released_structure != expected:
            errors.append(
                f"{scene_id}: composed prim structure changed: "
                f"expected={expected}, released={released_structure}"
            )
        scene_reports[scene_id] = {
            "layers": len(layers),
            "assets": len(assets),
            "unresolved": len(unresolved_paths),
            "external_layers": len(external_layers),
            "external_assets": len(external_assets),
            **released_structure,
        }
        print(
            f"  {scene_id}: layers={len(layers)} assets={len(assets)} "
            f"unresolved={len(unresolved_paths)} external="
            f"{len(external_layers) + len(external_assets)}",
            flush=True,
        )

    if errors:
        examples = "\n".join(f"  - {error}" for error in errors[:50])
        raise ReleaseError(f"release validation failed:\n{examples}")
    return {
        "authored_asset_paths": authored_asset_count,
        "mdl_modules": len(mdl_paths),
        "mdl_transitive_dependencies": len(mdl_dependencies),
        "private_path_files": 0,
        "symlinks": 0,
        "scenes": scene_reports,
    }


def write_release_documents(
    target_root: Path, scene_ids: Sequence[str], provenance: dict[str, dict[str, Any]]
) -> None:
    rows = "\n".join(f"- `{scene_id}`" for scene_id in scene_ids)
    processed_hashes = "\n".join(
        f"- `{scene_id}`: `{provenance[scene_id]['processed_sha256']}`" for scene_id in scene_ids
    )
    card = f"""# REAL Processed GRScenes {len(scene_ids)}-Scene Subset

This artifact contains {len(scene_ids)} experiment-specific GRScenes stages used by
REAL.  They are processed interaction scenes, not the original same-named
GRScenes stages.  Dependencies were first collected with InternUtopia's
`toolkits/grscenes_scripts/export_scenes.py`, then every authored USD asset path
was rewritten relative to this artifact and the complete closure was verified.

## Scenes

{rows}

The `MVUCSQAKTKJ5EAABAAAAABY8` entry preserves the experiment artifact with
the two unused demo camera prims removed.  Each entry point is
`scenes/<scene_id>_usd/scene.usd`.

## Processed input fingerprints

{processed_hashes}

## Layout

- `scenes/`: processed scene entry points
- `models/` and `Materials/`: scene dependency union
- `metadata/`: occupancy maps, furniture libraries, and generator metadata
- `mesa_required.txt`: separately downloaded objects used by the default demo
- `manifest.json` and `SHA256SUMS`: provenance and integrity records

## License and attribution

This is a derived data artifact.  The repository's MIT code license does not
replace the dataset terms.  GRScenes and the associated metadata are provided
under CC BY-NC-SA 4.0; preserve attribution, non-commercial restrictions, and
share-alike requirements.  Upstream project: InternRobotics/GRScenes.
"""
    (target_root / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    license_notice = """# Data license notice

This derived scene subset follows the upstream GRScenes data license:
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0).

License: https://creativecommons.org/licenses/by-nc-sa/4.0/
Upstream: https://huggingface.co/datasets/InternRobotics/GRScenes
"""
    (target_root / "LICENSE_DATA.md").write_text(license_notice, encoding="utf-8")
    (target_root / "mesa_required.txt").write_text(
        "\n".join(MESA_REQUIRED_OBJECTS) + "\n", encoding="utf-8"
    )


def iter_manifest_files(target_root: Path) -> Iterable[Path]:
    excluded = {"manifest.json", "SHA256SUMS"}
    return sorted(
        path
        for path in target_root.rglob("*")
        if path.is_file() and path.relative_to(target_root).as_posix() not in excluded
    )


def write_manifest(
    target_root: Path,
    scene_ids: Sequence[str],
    provenance: dict[str, dict[str, Any]],
    toolkit_exporter: Path,
    localization: dict[str, int],
    pruning: dict[str, int],
    validation: dict,
) -> dict:
    print("[6/7] Hashing release files and writing manifest", flush=True)
    file_entries = []
    total_bytes = 0
    for index, path in enumerate(iter_manifest_files(target_root), start=1):
        size = path.stat().st_size
        total_bytes += size
        file_entries.append(
            {
                "path": path.relative_to(target_root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
        if index % 1000 == 0:
            print(f"  hashed {index} files", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": (
            "real-processed-grscenes-7-v1"
            if tuple(scene_ids) == SCENE_IDS
            else "real-processed-grscenes-subset-v1"
        ),
        "scene_ids": list(scene_ids),
        "entry_point_template": "scenes/{scene_id}_usd/scene.usd",
        "internutopia_seed_stage_name": "start_result_raw.usd",
        "source_stage_name": next(iter(provenance.values()))["source_stage"],
        "scene_provenance": provenance,
        "special_processing": {
            **(
                {
                    ABY8_SCENE_ID: [
                        "removed /Root/Meshes/demo_cam_1",
                        "removed /Root/Meshes/demo_cam_2",
                    ]
                }
                if ABY8_SCENE_ID in scene_ids
                else {}
            )
        },
        "internutopia_exporter": {
            "repository_path": "toolkits/grscenes_scripts/export_scenes.py",
            "sha256": sha256_file(toolkit_exporter),
            "git_commit": get_git_commit(toolkit_exporter),
        },
        "release_builder": {
            "repository_path": "scripts/data/export_processed_grscenes.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": get_git_commit(Path(__file__).resolve()),
        },
        "localization": localization,
        "pruning": pruning,
        "validation": validation,
        "payload_file_count": len(file_entries),
        "payload_bytes": total_bytes,
        "files": file_entries,
    }
    manifest_path = target_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checksums = [f"{entry['sha256']}  {entry['path']}" for entry in file_entries]
    checksums.append(f"{sha256_file(manifest_path)}  manifest.json")
    (target_root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return manifest


def build_release(args: argparse.Namespace) -> Path:
    scene_ids = validate_scene_ids(args.scene_id or SCENE_IDS)
    source_root = require_directory(Path(args.source_root), "GRScenes source root")
    processed_root = require_directory(Path(args.processed_scenes_root), "processed scenes root")
    metadata_root = require_directory(Path(args.metadata_root), "metadata root")
    toolkit_exporter = require_file(Path(args.toolkit_exporter), "InternUtopia exporter")
    output = Path(args.output).expanduser().absolute()
    provenance = validate_inputs(
        source_root,
        processed_root,
        metadata_root,
        toolkit_exporter,
        output,
        scene_ids,
        args.source_stage_name,
        args.processed_stage_name,
    )
    print("[0/7] Fingerprinting processed input stage structure", flush=True)
    expected_structures = {}
    for scene_id in scene_ids:
        structure = stage_structure(processed_root / f"{scene_id}_usd" / args.processed_stage_name)
        expected_structures[scene_id] = structure
        provenance[scene_id]["processed_structure"] = structure

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise ReleaseError(f"stale partial output exists: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)

    try:
        run_internutopia_exporter(toolkit_exporter, source_root, partial, scene_ids)
        select_processed_stages(partial, processed_root, scene_ids, args.processed_stage_name)
        copy_runtime_metadata(partial, metadata_root, scene_ids)
        localization = localize_usd_asset_paths(partial)
        pruning = prune_to_scene_closure(partial, scene_ids)
        write_release_documents(partial, scene_ids, provenance)
        validation = verify_release(partial, scene_ids, expected_structures)
        manifest = write_manifest(
            partial,
            scene_ids,
            provenance,
            toolkit_exporter,
            localization,
            pruning,
            validation,
        )
        print("[7/7] Rechecking generated checksums", flush=True)
        checksum_result = subprocess.run(
            ["sha256sum", "--check", "SHA256SUMS"],
            cwd=partial,
            check=False,
            capture_output=True,
            text=True,
        )
        if checksum_result.returncode != 0:
            raise ReleaseError(f"checksum verification failed: {checksum_result.stderr}")
        if output.exists() or output.is_symlink():
            raise ReleaseError(f"output appeared during build; refusing to replace it: {output}")
        partial.rename(output)
        print(
            f"READY: {len(scene_ids)}/{len(scene_ids)} scenes, "
            f"{manifest['payload_file_count']} payload files, "
            f"{manifest['payload_bytes']} bytes -> {output}",
            flush=True,
        )
        return output
    except Exception:
        # Preserve the real validation/export exception.  Some distributed
        # filesystems can transiently report ENOTEMPTY while metadata settles.
        if partial.exists() and not args.keep_failed:
            try:
                shutil.rmtree(partial)
            except OSError as cleanup_error:
                print(
                    f"WARNING: failed to remove partial output {partial}: {cleanup_error}",
                    file=sys.stderr,
                )
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export REAL's processed GRScenes subset with InternUtopia and "
            "make every dependency portable"
        )
    )
    parser.add_argument("--source-root", required=True, help="full GRScenes home_scenes root")
    parser.add_argument(
        "--processed-scenes-root",
        required=True,
        help="root containing <scene_id>_usd/scene.usd experiment artifacts",
    )
    parser.add_argument("--metadata-root", required=True, help="REAL assets/metadata root")
    parser.add_argument(
        "--toolkit-exporter",
        required=True,
        help="InternUtopia toolkits/grscenes_scripts/export_scenes.py",
    )
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument(
        "--scene-id",
        action="append",
        help="scene id to export; repeat as needed (defaults to REAL's seven scenes)",
    )
    parser.add_argument("--source-stage-name", default="start_result_interaction_noMDL_move.usd")
    parser.add_argument("--processed-stage-name", default="scene.usd")
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="retain the partial output for debugging when a validation gate fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        build_release(args)
    except (OSError, ReleaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
