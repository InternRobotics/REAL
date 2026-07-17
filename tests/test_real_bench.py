import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from shutil import copytree

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

from mcp_server.config import load_task_config  # noqa: E402
from real_bench import RealBenchValidationError, load_real_bench  # noqa: E402


class RealBenchReleaseTest(unittest.TestCase):
    def test_each_episode_is_a_direct_eval_server_config(self):
        task_root = REPO_ROOT / "benchmark" / "tasks"
        paths = sorted(task_root.glob("*/*.yaml"))

        self.assertEqual(len(paths), 241)
        for path in paths:
            config = load_task_config(path)
            self.assertEqual(len(config["episodes"]), 1, str(path))
            episode = config["episodes"][0]
            self.assertEqual(path.parent.name, episode["family"], str(path))
            self.assertEqual(path.stem, episode["task_id"], str(path))
            self.assertEqual(set(episode["placements"]), set(config["objects"]), str(path))

    def test_benchmark_uses_yaml_as_its_only_structured_format(self):
        benchmark_root = REPO_ROOT / "benchmark"
        yaml_paths = sorted(benchmark_root.rglob("*.yaml"))

        self.assertTrue(yaml_paths)
        self.assertFalse(list(benchmark_root.rglob("*.json")))
        for path in yaml_paths:
            with path.open(encoding="utf-8") as stream:
                self.assertIsNotNone(yaml.load(stream, Loader=YAML_LOADER), str(path))

    def test_complete_bundle_loads_241_unique_tasks(self):
        tasks = load_real_bench()

        self.assertEqual(len(tasks), 241)
        self.assertEqual(
            Counter(task["family"] for task in tasks),
            {"FDP": 72, "FODP": 56, "FDO": 48, "SUL": 65},
        )
        self.assertEqual(len({task["benchmark_task_id"] for task in tasks}), 241)
        self.assertEqual([task["global_index"] for task in tasks], list(range(241)))

    def test_family_filter_validates_bundle_then_selects_family(self):
        tasks = load_real_bench(family="sul")

        self.assertEqual(len(tasks), 65)
        self.assertEqual({task["family"] for task in tasks}, {"SUL"})

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(RealBenchValidationError, "Unknown family"):
            load_real_bench(family="unknown")

    def test_release_has_no_machine_specific_paths(self):
        benchmark_root = REPO_ROOT / "benchmark"
        markers = ("/cpfs/", "/shared/", "/home/", "/mnt/")

        for path in benchmark_root.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                self.assertFalse(
                    any(marker in text for marker in markers),
                    f"machine-specific path found in {path}",
                )

    def test_release_has_only_canonical_per_episode_views(self):
        benchmark_root = REPO_ROOT / "benchmark"
        manifest = yaml.load(
            (benchmark_root / "manifest.yaml").read_text(encoding="utf-8"),
            Loader=YAML_LOADER,
        )

        self.assertEqual(manifest.get("status"), "official")
        self.assertNotIn("provenance", manifest)
        self.assertNotIn("warning", manifest)
        self.assertEqual(
            {path.name for path in benchmark_root.iterdir()},
            {"README.md", "manifest.yaml", "mesa_required.txt", "tasks"},
        )
        task_root = benchmark_root / "tasks"
        self.assertEqual(
            {path.name for path in task_root.iterdir() if path.is_dir()},
            set(manifest["family_order"]),
        )
        self.assertEqual(len(list(task_root.glob("*/*.yaml"))), manifest["total_tasks"])

    def test_task_objects_are_consistent_in_each_episode(self):
        for task in load_real_bench():
            task_id = task["benchmark_task_id"]
            placement_ids = Counter(
                placement["original_id"] for placement in task["placements"].values()
            )
            initial_ids = Counter(
                object_id
                for state in task["initial_world_graph"].values()
                for object_id in state.get("content", [])
            )
            metadata_ids = set(task.get("obj_meta", {})) | set(task.get("obj_distractor_meta", {}))

            self.assertEqual(initial_ids, placement_ids, task_id)
            self.assertTrue(set(placement_ids).issubset(metadata_ids), task_id)

            if "final_positions" in task:
                final_ids = Counter(
                    position["original_id"] for position in task["final_positions"].values()
                )
                self.assertEqual(final_ids, placement_ids, task_id)

            target_ids = list(task.get("obj_meta", {}))
            self.assertEqual(len(target_ids), 1, task_id)
            target_id = target_ids[0]
            source = task["src"]
            destination = task["dest"]
            self.assertIn(target_id, task["initial_world_graph"][source]["content"], task_id)
            self.assertNotIn(target_id, task["goal_world_graph"][source]["content"], task_id)
            self.assertIn(target_id, task["goal_world_graph"][destination]["content"], task_id)

    def test_loader_rejects_episode_object_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            copied_root = Path(temporary_dir) / "benchmark"
            copytree(REPO_ROOT / "benchmark", copied_root)

            episode_path = copied_root / "tasks" / "SUL" / "verify_task_1.yaml"
            config = yaml.load(episode_path.read_text(encoding="utf-8"), Loader=YAML_LOADER)
            episode = config["episodes"][0]
            episode["initial_world_graph"][episode["src"]]["content"][0] = "wrong-object"
            episode_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RealBenchValidationError, "initial world graph"):
                load_real_bench(copied_root)

    def test_module_cli_validates_complete_bundle(self):
        result = subprocess.run(
            [sys.executable, "-m", "real_bench"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Loaded and validated 241 REAL-Bench tasks", result.stdout)
        self.assertIn("FDP=72, FODP=56, FDO=48, SUL=65", result.stdout)


if __name__ == "__main__":
    unittest.main()
