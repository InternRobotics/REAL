#!/usr/bin/env python3
"""Patch ms-swift 3.12.4 to route NGRPO advantages to this repository."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import py_compile
import shutil
import tempfile
from pathlib import Path

SUPPORTED_VERSION = "3.12.4"
ARGUMENTS_RELATIVE = Path("trainers/arguments.py")
RLHF_ARGS_RELATIVE = Path("llm/argument/rlhf_args.py")
TRAINER_RELATIVE = Path("trainers/rlhf_trainer/grpo_trainer.py")

ARGUMENTS_CLEAN = """    # RLOO, REINFORCE++
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
"""
ARGUMENTS_PATCHED = """    # RLOO, REINFORCE++, NGRPO
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus', 'ngrpo'] = 'grpo'
    # NGRPO virtual max reward (None uses group max plus a margin)
    ngrpo_virtual_max_reward: Optional[float] = None
"""

NGRPO_BRANCH = """            if self.advantage_estimator == 'ngrpo':
                from rl_runtime.ngrpo import compute_ngrpo_advantages
                advantages = compute_ngrpo_advantages(
                    grouped_rewards, self.ngrpo_virtual_max_reward)
            elif self.advantage_estimator == 'rloo':
"""
NGRPO_BRANCH_START = "            if self.advantage_estimator == 'ngrpo':\n"
RLOO_BRANCH = "            if self.advantage_estimator == 'rloo':\n"
RLOO_AFTER_NGRPO = "            elif self.advantage_estimator == 'rloo':\n"

NORMALIZE_CLEAN = "            if self.advantage_estimator == 'reinforce_plus_plus':\n"
NORMALIZE_PATCHED = """            if self.advantage_estimator == 'ngrpo':
                pass
            elif self.advantage_estimator == 'reinforce_plus_plus':
"""

LIGER_ASSERT_CLEAN = "        assert self.advantage_estimator == 'grpo'\n"
LIGER_ASSERT_PATCHED = "        assert self.advantage_estimator in ('grpo', 'ngrpo')\n"

INIT_ANCHOR = "        self.dynamic_sample = args.dynamic_sample\n"
INIT_PATCHED = """        self.ngrpo_virtual_max_reward = getattr(
            args, 'ngrpo_virtual_max_reward', None)
        self.dynamic_sample = args.dynamic_sample
"""


class PatchError(RuntimeError):
    """Raised when an expected ms-swift source anchor cannot be found."""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise PatchError(f"expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def patch_arguments(source: str) -> str:
    if "ngrpo_virtual_max_reward: Optional[float]" in source:
        return source
    return replace_once(source, ARGUMENTS_CLEAN, ARGUMENTS_PATCHED, "GRPO arguments")


def patch_rlhf_args(source: str) -> str:
    default_clean = "if self.advantage_estimator == 'grpo':"
    default_patched = "if self.advantage_estimator in ['grpo', 'ngrpo']:"
    if source.count(default_patched) < 2:
        if source.count(default_clean) != 2:
            raise PatchError("cannot locate both GRPO default-selection branches")
        source = source.replace(default_clean, default_patched, 2)

    liger_clean = (
        "if self.advantage_estimator != 'grpo':\n"
        "                raise ValueError('Liger loss currently only support grpo advantage estimator')"
    )
    liger_patched = (
        "if self.advantage_estimator not in ['grpo', 'ngrpo']:\n"
        "                raise ValueError('Liger loss currently only supports grpo/ngrpo advantage estimators')"
    )
    legacy_liger_patched = (
        "if self.advantage_estimator not in ['grpo', 'ngrpo']:\n"
        "                raise ValueError('Liger loss currently only support grpo/ngrpo advantage estimator')"
    )
    if liger_patched not in source and legacy_liger_patched not in source:
        source = replace_once(source, liger_clean, liger_patched, "Liger validation")
    return source


def replace_advantage_branch(source: str) -> str:
    if "from rl_runtime.ngrpo import compute_ngrpo_advantages" in source:
        return source

    ngrpo_start = source.find(NGRPO_BRANCH_START)
    if ngrpo_start >= 0:
        rloo_start = source.find(RLOO_AFTER_NGRPO, ngrpo_start)
        if rloo_start < 0:
            raise PatchError("legacy inline NGRPO branch has no following RLOO branch")
        rloo_end = rloo_start + len(RLOO_AFTER_NGRPO)
        return source[:ngrpo_start] + NGRPO_BRANCH + source[rloo_end:]

    if RLOO_BRANCH not in source:
        raise PatchError("cannot locate the grouped RLOO advantage branch")
    return source.replace(RLOO_BRANCH, NGRPO_BRANCH, 1)


def patch_trainer(source: str) -> str:
    source = replace_advantage_branch(source)
    first_ngrpo = source.find(NGRPO_BRANCH_START)
    second_ngrpo = source.find(NGRPO_BRANCH_START, first_ngrpo + 1)
    if second_ngrpo < 0:
        if NORMALIZE_CLEAN not in source:
            raise PatchError("cannot locate the grouped advantage normalization branch")
        source = source.replace(NORMALIZE_CLEAN, NORMALIZE_PATCHED, 1)
    if LIGER_ASSERT_PATCHED not in source:
        source = replace_once(source, LIGER_ASSERT_CLEAN, LIGER_ASSERT_PATCHED, "Liger assertion")
    if "self.ngrpo_virtual_max_reward = getattr(" not in source:
        source = replace_once(source, INIT_ANCHOR, INIT_PATCHED, "trainer initialization")
    return source


PATCHERS = {
    ARGUMENTS_RELATIVE: patch_arguments,
    RLHF_ARGS_RELATIVE: patch_rlhf_args,
    TRAINER_RELATIVE: patch_trainer,
}


def is_rerouted(swift_root: Path) -> bool:
    arguments = (swift_root / ARGUMENTS_RELATIVE).read_text(encoding="utf-8")
    rlhf_args = (swift_root / RLHF_ARGS_RELATIVE).read_text(encoding="utf-8")
    trainer = (swift_root / TRAINER_RELATIVE).read_text(encoding="utf-8")
    return all(
        (
            "ngrpo_virtual_max_reward: Optional[float]" in arguments,
            rlhf_args.count("if self.advantage_estimator in ['grpo', 'ngrpo']:") >= 2,
            "if self.advantage_estimator not in ['grpo', 'ngrpo']:" in rlhf_args,
            "from rl_runtime.ngrpo import compute_ngrpo_advantages" in trainer,
            LIGER_ASSERT_PATCHED in trainer,
            "self.ngrpo_virtual_max_reward = getattr(" in trainer,
        )
    )


def locate_swift_root() -> Path:
    spec = importlib.util.find_spec("swift")
    if spec is None or not spec.submodule_search_locations:
        raise PatchError("cannot locate the installed swift package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def apply_patch(swift_root: Path) -> list[Path]:
    changed = []
    for relative_path, patcher in PATCHERS.items():
        path = swift_root / relative_path
        source = path.read_text(encoding="utf-8")
        patched = patcher(source)
        if source == patched:
            continue
        backup = path.with_suffix(path.suffix + ".pre_ngrpo_reroute")
        if not backup.exists():
            shutil.copy2(path, backup)
        atomic_write(path, patched)
        py_compile.compile(str(path), doraise=True)
        changed.append(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swift-root",
        type=Path,
        help="swift package directory; defaults to the active interpreter installation",
    )
    parser.add_argument("--apply", action="store_true", help="apply the reroute in place")
    args = parser.parse_args()

    swift_root = (args.swift_root or locate_swift_root()).resolve()
    if args.swift_root is None:
        installed_version = importlib.metadata.version("ms-swift")
        if installed_version != SUPPORTED_VERSION:
            raise PatchError(f"ms-swift {installed_version} is installed; expected {SUPPORTED_VERSION}")

    if not args.apply:
        if is_rerouted(swift_root):
            print(f"NGRPO reroute is installed: {swift_root}")
            return 0
        print(f"NGRPO reroute is not installed: {swift_root}")
        print("rerun with --apply after reviewing the target interpreter")
        return 1

    changed = apply_patch(swift_root)
    if not is_rerouted(swift_root):
        raise PatchError("patch completed but reroute verification failed")
    if changed:
        print("patched files:")
        for path in changed:
            print(f"  {path}")
    else:
        print(f"NGRPO reroute already installed: {swift_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
