import importlib
import os
import tempfile
import unittest
from unittest import mock


with mock.patch.dict(
    os.environ,
    {"TASK_SOURCE_PATH": os.devnull, "OUTPUT_PATH": tempfile.gettempdir()},
):
    verify_proc = importlib.import_module("proc_datagen.verify_proc")


class UsdPrimComponentNameTests(unittest.TestCase):
    def test_sanitizes_digit_prefix_and_invalid_characters(self):
        used_names = set()

        component = verify_proc._allocate_usd_prim_component("79a/object-on table_t0", used_names)

        self.assertRegex(component, r"^[A-Za-z_][A-Za-z0-9_]*$")
        self.assertTrue(component.startswith("obj_79a_object_on_table_t0"))
        self.assertEqual({component}, used_names)

    def test_keeps_valid_names(self):
        self.assertEqual(
            "mug_on_table_t3",
            verify_proc._allocate_usd_prim_component("mug_on_table_t3", set()),
        )

    def test_resolves_sanitization_collisions_deterministically(self):
        def allocate_sequence():
            used_names = set()
            return (
                verify_proc._allocate_usd_prim_component("mug-on-table_t3", used_names),
                verify_proc._allocate_usd_prim_component("mug_on_table_t3", used_names),
            )

        first_run = allocate_sequence()
        second_run = allocate_sequence()

        self.assertEqual(first_run, second_run)
        self.assertEqual(("mug_on_table_t3", "mug_on_table_t3_2"), first_run)


if __name__ == "__main__":
    unittest.main()
