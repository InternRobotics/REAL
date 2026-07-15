#!/usr/bin/env python3
"""Validate a standalone REAL RL runtime before launching expensive jobs."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_runtime.real_contract import REAL_UPSTREAM_COMMIT, REAL_UPSTREAM_URL  # noqa: E402

REQUIRED_DISTRIBUTIONS = (
    "accelerate",
    "deepspeed",
    "gymnasium",
    "httpx",
    "matplotlib",
    "mcp",
    "modelscope",
    "ms-swift",
    "numpy",
    "opencv-python-headless",
    "peft",
    "Pillow",
    "qwen-vl-utils",
    "requests",
    "tensorboard",
    "torch",
    "transformers",
    "trl",
    "vllm",
)


def check_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"{label} not found: {path}")


def check_dataset(path: Path, errors: list[str]) -> None:
    check_file(path, "dataset", errors)
    if errors:
        return
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"dataset is not valid JSON: {exc}")
        return
    if not isinstance(records, list) or not records:
        errors.append("dataset must be a non-empty JSON list")
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            errors.append(f"dataset record {index} must contain a messages list")
            break


def check_model(path: Path, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"model directory not found: {path}")
        return
    check_file(path / "config.json", "model config", errors)
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weights:
        errors.append(f"model weights not found in: {path}")


def check_packages(errors: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Python distribution is not installed: {distribution}")
    return versions


def check_cuda(errors: list[str]) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        errors.append(f"cannot import torch: {exc}")
        return {}
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false")
        return {"torch": torch.__version__}
    return {
        "torch": torch.__version__,
        "visible_devices": torch.cuda.device_count(),
        "device_0": torch.cuda.get_device_name(0),
    }


def check_reward_contract(errors: list[str]) -> None:
    """Verify the local Gym adapter and installed ms-swift reward handoff."""
    try:
        from rl_runtime.environment import MCPGymEnv

        class FixedRewardEnv:
            async def step(self, action):
                return "next observation", 1.25, True, {"source": "fixed"}

            async def disconnect(self):
                return None

        gym_env = MCPGymEnv(env_config={})
        gym_env._env = FixedRewardEnv()
        observation, reward, terminated, truncated, info = gym_env.step("action")
        gym_env.close()
        expected = ("next observation", 1.25, True, False, 1.25)
        actual = (observation, reward, terminated, truncated, info.get("reward"))
        if actual != expected:
            errors.append(f"MCPGymEnv did not forward the fixed reward: {actual!r}")
    except Exception as exc:
        errors.append(f"cannot validate MCPGymEnv reward forwarding: {exc!r}")

    try:
        from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer

        trainer_source = Path(inspect.getfile(GRPOTrainer)).read_text(encoding="utf-8")
        required_fragments = (
            "if self.use_gym_env:",
            "inp['rollout_infos']['total_reward']",
            "reward_from_gym",
        )
        missing = [fragment for fragment in required_fragments if fragment not in trainer_source]
        if missing:
            errors.append(f"installed ms-swift does not expose the expected Gym reward handoff: {missing!r}")
    except Exception as exc:
        errors.append(f"cannot inspect ms-swift Gym reward handoff: {exc!r}")

    try:
        scheduler_source = (ROOT / "rl_runtime" / "scheduler.py").read_text(encoding="utf-8")
        required_fragments = (
            "total_reward += reward",
            '"total_reward": float(total_reward)',
            '"reward_source": "mcp_step_sum"',
        )
        missing = [fragment for fragment in required_fragments if fragment not in scheduler_source]
        if missing:
            errors.append(f"scheduler does not preserve the MCP reward path: {missing!r}")
    except OSError as exc:
        errors.append(f"cannot inspect scheduler reward handoff: {exc!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DATASET_PATH", ROOT / "scripts/grpo_gym_dataset.json")),
    )
    parser.add_argument("--cuda", action="store_true", help="require an accessible CUDA device")
    parser.add_argument("--import-plugin", action="store_true")
    parser.add_argument("--check-reward-contract", action="store_true")
    parser.add_argument("--skip-packages", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    check_dataset(args.dataset.expanduser().resolve(), errors)
    if args.model:
        check_model(args.model.expanduser().resolve(), errors)

    versions = {} if args.skip_packages else check_packages(errors)
    cuda = check_cuda(errors) if args.cuda else {}

    if args.check_reward_contract:
        check_reward_contract(errors)

    if args.import_plugin:
        try:
            import rl_runtime.swift_plugin  # noqa: F401
        except Exception as exc:
            errors.append(f"cannot import ms-swift plugin: {exc!r}")

    report = {
        "ok": not errors,
        "real_upstream": REAL_UPSTREAM_URL,
        "real_commit": REAL_UPSTREAM_COMMIT,
        "dataset": str(args.dataset),
        "model": str(args.model) if args.model else None,
        "packages": versions,
        "cuda": cuda,
        "reward_contract_checked": args.check_reward_contract,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
