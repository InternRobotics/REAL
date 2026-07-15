import asyncio
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from proc_datagen import task_generator  # noqa: E402


class _ProgressBar:
    def __init__(self):
        self.count = 0

    def update(self, count=1):
        self.count += count


def _make_articulation_tasks():
    generator = object.__new__(task_generator.ArticulationPnPGenerator)
    generator.scene_furniture = {
        "scene_a": [
            {
                "name": "table_1",
                "room_name": "kitchen",
                "functionals": [],
            },
            {
                "name": "fridge_1",
                "room_name": "kitchen",
                "functionals": ["door"],
            },
        ]
    }
    generator.max_objects_per_pair = 1
    generator.asset_library = {
        "asset_1": {
            "category": "mug",
            "usd_path": "asset_1.usd",
            "size": [0.1, 0.1, 0.2],
        }
    }
    generator.stats = defaultdict(int)
    generator._get_objects_for_pair = lambda *_args: [("mug", "asset_1")]
    generator._get_furniture_caption = lambda furniture: furniture["name"]
    generator._desc_basic = lambda category, src, dest: f"Move {category} from {src} to {dest}."
    generator._get_distractors_non_artic = lambda *_args: []
    generator._select_obj_distractors = lambda *_args: []

    def _check_layout(task):
        position_x = 1.0 if task["src"] == "table_1" else 2.0
        return True, {
            f"asset_1_on_{task['src']}": {
                "original_id": "asset_1",
                "furniture": task["src"],
                "position": [position_x, 0.0, 1.0],
            }
        }

    generator._check_layout = _check_layout
    return generator.generate_tasks_for_scene("scene_a")


class ArticulationExportContractTest(unittest.TestCase):
    def test_generator_uses_canonical_type_and_preserves_subtype(self):
        tasks = _make_articulation_tasks()

        self.assertEqual({task["task_type"] for task in tasks}, {"articulation"})
        self.assertEqual(
            {task["articulation_subtype"] for task in tasks},
            {"store", "retrieve"},
        )

    def test_export_writes_one_articulation_file_with_episode_subtypes(self):
        tasks = _make_articulation_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            asset_library = tmp_path / "assets.json"
            asset_library.write_text(json.dumps({}), encoding="utf-8")

            task_generator.export_tasks_to_yaml(tasks, str(tmp_path / "output"), str(asset_library))

            output = tmp_path / "output" / "scene_a" / "articulation.yaml"
            self.assertTrue(output.is_file())
            document = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertEqual(document["task_type"], "articulation")
            self.assertEqual(
                {episode["articulation_subtype"] for episode in document["episodes"]},
                {"store", "retrieve"},
            )


class MergeContractTest(unittest.TestCase):
    def test_merge_rewrites_placement_keys_with_object_keys(self):
        scenes = [
            "MVUCSQAKTKJ5EAABAAAAABQ8",
            "MVUCSQAKTKJ5EAABAAAAAAQ8",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_source = tmp_path / "source"
            output_root = tmp_path / "results"

            for index, scene_id in enumerate(scenes):
                source_dir = task_source / scene_id
                passed_dir = output_root / "distractor" / scene_id
                source_dir.mkdir(parents=True)
                passed_dir.mkdir(parents=True)
                (source_dir / "distractor.yaml").write_text(
                    yaml.safe_dump({"episodes": [{"task_id": f"task_{index}"}]}),
                    encoding="utf-8",
                )
                (passed_dir / "physics_passed.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "objects": {
                                "obj_0": {
                                    "original_id": f"asset_{index}",
                                    "position": [float(index), 0.0, 1.0],
                                }
                            },
                            "episodes": [
                                {
                                    "task_id": f"task_{index}",
                                    "placements": {
                                        "obj_0": {
                                            "original_id": f"asset_{index}",
                                            "furniture": "table_1",
                                        }
                                    },
                                }
                            ],
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )

            env = os.environ.copy()
            env.update(
                {
                    "TASK_SRC_DIR": str(task_source),
                    "OUTPUT_ROOT": str(output_root),
                    "PYTHON_BIN": sys.executable,
                }
            )
            subprocess.run(
                [str(REPO_ROOT / "scripts/filter/batch_filter_proc.sh"), "--stage", "merge"],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            merged_path = output_root / "distractor" / "physics_valid.yaml"
            merged = yaml.safe_load(merged_path.read_text(encoding="utf-8"))
            object_keys = set(merged["objects"])
            self.assertEqual(len(object_keys), 2)
            for episode in merged["episodes"]:
                placement_keys = set(episode["placements"])
                self.assertTrue(placement_keys)
                self.assertLessEqual(placement_keys, object_keys)
                self.assertTrue(
                    all(key.startswith(f"{episode['scene_id']}_") for key in placement_keys)
                )


class PolishContractTest(unittest.TestCase):
    def test_polish_uses_openai_model_environment_variable(self):
        calls = []

        class _Completions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="Polished text.")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        class _Client:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(completions=_Completions())

        bar = _ProgressBar()
        with (
            mock.patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-polish-model"},
            ),
            mock.patch.object(task_generator, "_AsyncOpenAI", _Client, create=True),
        ):
            result = asyncio.run(task_generator._polish_batch_async(["Raw text."], bar))

        self.assertEqual(result, ["Polished text."])
        self.assertEqual(calls[0]["model"], "test-polish-model")
        self.assertEqual(bar.count, 1)

    def test_polish_treats_blank_optional_env_as_defaults(self):
        constructor_calls = []
        completion_calls = []

        class _Completions:
            async def create(self, **kwargs):
                completion_calls.append(kwargs)
                message = types.SimpleNamespace(content="Polished text.")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        class _Client:
            def __init__(self, **kwargs):
                constructor_calls.append(kwargs)
                self.chat = types.SimpleNamespace(completions=_Completions())

        with (
            mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_API_BASE_URL": "",
                    "OPENAI_MODEL": "",
                },
            ),
            mock.patch.object(task_generator, "_AsyncOpenAI", _Client, create=True),
        ):
            result = asyncio.run(task_generator._polish_batch_async(["Raw text."], _ProgressBar()))

        self.assertEqual(result, ["Polished text."])
        self.assertEqual(constructor_calls[0]["base_url"], "https://api.openai.com/v1")
        self.assertEqual(completion_calls[0]["model"], "gpt-4o-mini")

    def test_polish_raises_after_retry_exhaustion(self):
        attempts = 0

        class _Completions:
            async def create(self, **_kwargs):
                nonlocal attempts
                attempts += 1
                raise ConnectionError("endpoint unavailable")

        class _Client:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(completions=_Completions())

        async def _no_sleep(_delay):
            return None

        bar = _ProgressBar()
        with (
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            mock.patch.object(task_generator, "_AsyncOpenAI", _Client, create=True),
            mock.patch.object(task_generator._asyncio, "sleep", _no_sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
                asyncio.run(task_generator._polish_batch_async(["Raw text."], bar))

        self.assertEqual(attempts, 3)
        self.assertEqual(bar.count, 1)


if __name__ == "__main__":
    unittest.main()
