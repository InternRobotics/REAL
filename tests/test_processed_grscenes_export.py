import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/data/export_processed_grscenes.py"
SPEC = importlib.util.spec_from_file_location("export_processed_grscenes", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


class ProcessedSceneExportTest(unittest.TestCase):
    def test_default_release_has_exactly_seven_unique_scenes(self):
        self.assertEqual(len(exporter.SCENE_IDS), 7)
        self.assertEqual(len(set(exporter.SCENE_IDS)), 7)
        self.assertEqual(set(exporter.EXPECTED_PROCESSED_SHA256), set(exporter.SCENE_IDS))
        self.assertEqual(set(exporter.EXPECTED_RAW_SEED_SHA256), set(exporter.SCENE_IDS))
        self.assertIn(exporter.ABY8_SCENE_ID, exporter.SCENE_IDS)

    def test_scene_ids_reject_path_traversal(self):
        for scene_id in ("../scene", "scene/name", "scene_name"):
            with self.subTest(scene_id=scene_id):
                with self.assertRaises(exporter.ReleaseError):
                    exporter.validate_scene_ids([scene_id])

    def test_asset_locator_uses_explicit_export_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layer = root / "scenes/scene_usd/scene.usd"
            model = root / "models/chair/instance.usd"
            material = root / "Materials/Textures/color.png"
            default_texture = root / "Materials/Textures/T_Default_Material_Grid_N.png"
            for path in (layer, model, material, default_texture):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            locator = exporter.AssetLocator(root)
            resolved, method = locator.resolve(layer, "/private/archive/models/chair/instance.usd")
            self.assertEqual(resolved, model.resolve())
            self.assertEqual(method, "marker:models")

            resolved, method = locator.resolve(
                layer,
                "/private/archive/models/chair/Materials/Textures/color.png",
            )
            self.assertEqual(resolved, material.resolve())
            self.assertEqual(method, "marker:Materials")

            resolved, method = locator.resolve(
                layer,
                "/private/archive/models/chair/Textures/T_Default_Material_Grid_N.png",
            )
            self.assertEqual(resolved, default_texture.resolve())
            self.assertEqual(method, "shared-default-texture")

    def test_asset_locator_does_not_guess_unknown_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layer = root / "scenes/scene_usd/scene.usd"
            unrelated = root / "Materials/Textures/unknown.png"
            for path in (layer, unrelated):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            resolved, reason = exporter.AssetLocator(root).resolve(layer, "/private/unknown.png")
            self.assertIsNone(resolved)
            self.assertEqual(reason, "not-found")

    def test_mdl_dependency_scan_follows_local_modules_and_textures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materials = root / "Materials"
            textures = materials / "Textures"
            textures.mkdir(parents=True)
            top = materials / "Top.mdl"
            child = materials / "Child.mdl"
            texture = textures / "color.jpg"
            top.write_text(
                'using .::Child import *;\ntexture_2d("./Textures/color.jpg", gamma_default);\n',
                encoding="utf-8",
            )
            child.write_text("import ::math::*;\n", encoding="utf-8")
            texture.write_bytes(b"image")

            dependencies = exporter.collect_mdl_dependencies([top], root)
            self.assertEqual(dependencies, {child.resolve(), texture.resolve()})

    def test_mdl_dependency_scan_rejects_unknown_missing_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materials = root / "Materials"
            materials.mkdir()
            top = materials / "Top.mdl"
            top.write_text("import ::MissingProjectModule::*;\n", encoding="utf-8")
            with self.assertRaises(exporter.ReleaseError):
                exporter.collect_mdl_dependencies([top], root)

    def test_private_marker_scan_handles_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layer.usd"
            path.write_bytes(b"x" * (1024 * 1024 - 3) + b"/cp" + b"fs/secret")
            self.assertTrue(exporter._contains_private_marker(path))

    def test_release_documents_list_external_demo_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_id = exporter.SCENE_IDS[0]
            exporter.write_release_documents(
                root,
                [scene_id],
                {scene_id: {"processed_sha256": "a" * 64}},
            )
            required = (root / "mesa_required.txt").read_text().splitlines()
            self.assertEqual(required, list(exporter.MESA_REQUIRED_OBJECTS))


if __name__ == "__main__":
    unittest.main()
