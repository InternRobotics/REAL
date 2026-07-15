"""Regression tests for the public MCP interaction and perception contracts."""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


def test_repo_path_expands_mesatask_environment_variable(monkeypatch, tmp_path):
    from mcp_server.config import resolve_repo_path

    monkeypatch.setenv("MESATASK_USD_ROOT", str(tmp_path))

    assert resolve_repo_path("${MESATASK_USD_ROOT}/object.usd") == (tmp_path / "object.usd")


def test_repo_path_rejects_unset_environment_variable(monkeypatch):
    from mcp_server.config import resolve_repo_path

    monkeypatch.delenv("MESATASK_USD_ROOT", raising=False)

    with pytest.raises(ValueError, match="MESATASK_USD_ROOT"):
        resolve_repo_path("$MESATASK_USD_ROOT/object.usd")


def test_ask_tool_schema_matches_generated_plan_arguments():
    from mcp_server.tools import MCP_TOOLS, OPENAI_TOOLS

    mcp_ask = next(tool for tool in MCP_TOOLS if tool.name == "ask")
    assert mcp_ask.inputSchema["required"] == [
        "target_category",
        "target_description",
    ]
    assert set(mcp_ask.inputSchema["properties"]) == {
        "target_category",
        "target_description",
    }

    openai_ask = next(
        tool["function"] for tool in OPENAI_TOOLS if tool["function"]["name"] == "ask"
    )
    assert openai_ask["parameters"] == mcp_ask.inputSchema


@pytest.mark.parametrize(
    "execution_plan",
    [
        [
            {
                "action": "ask",
                "args": {
                    "target_category": "tray",
                    "target_description": "A dark rectangular tray.",
                },
            }
        ],
        ["navigate to table_1", "ask tray - A dark rectangular tray."],
    ],
)
def test_simulated_user_context_is_derived_from_task_metadata(execution_plan):
    from mcp_server.interaction import build_simulated_user_context

    context = build_simulated_user_context(
        {"execution_plan": execution_plan},
        fallback_category="wrong fallback",
    )

    assert context == {
        "target_category": "tray",
        "target_description": "A dark rectangular tray.",
        "source": "task_metadata",
    }


def _install_actions_import_stubs(monkeypatch):
    pxr = types.ModuleType("pxr")
    pxr.UsdPhysics = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    internutopia = types.ModuleType("internutopia")
    core = types.ModuleType("internutopia.core")
    util = types.ModuleType("internutopia.core.util")
    omni_usd_util = types.ModuleType("internutopia.core.util.omni_usd_util")
    omni_usd_util.compute_path_bbox = lambda _path: None
    monkeypatch.setitem(sys.modules, "internutopia", internutopia)
    monkeypatch.setitem(sys.modules, "internutopia.core", core)
    monkeypatch.setitem(sys.modules, "internutopia.core.util", util)
    monkeypatch.setitem(
        sys.modules,
        "internutopia.core.util.omni_usd_util",
        omni_usd_util,
    )

    extension = types.ModuleType("internutopia_extension")
    extension_utils = types.ModuleType("internutopia_extension.utils")
    camera_utils = types.ModuleType("internutopia_extension.utils.camera_utils")
    camera_utils.track_object = lambda *_args, **_kwargs: None
    camera_utils.calculate_look_at_quaternion_fixed_camera = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "internutopia_extension", extension)
    monkeypatch.setitem(sys.modules, "internutopia_extension.utils", extension_utils)
    monkeypatch.setitem(
        sys.modules,
        "internutopia_extension.utils.camera_utils",
        camera_utils,
    )

    mcp_env = types.ModuleType("mcp_server.mcp_env")
    mcp_env.nav_manager = None
    mcp_env.camera_prim = None
    mcp_env.object_per_room = {}
    mcp_env.furniture_names = []
    mcp_env.furniture_prims = []
    monkeypatch.setitem(sys.modules, "mcp_server.mcp_env", mcp_env)

    perception = types.ModuleType("mcp_server.perception_utils")
    perception.find_objects = lambda *_args, **_kwargs: None
    perception.highlight_receptacles = lambda *_args, **_kwargs: None
    perception.render_persisted_markers = lambda *_args, **_kwargs: None
    perception.get_rgb_image = lambda: None
    monkeypatch.setitem(sys.modules, "mcp_server.perception_utils", perception)


def test_ask_dispatch_returns_canonical_task_metadata(monkeypatch):
    _install_actions_import_stubs(monkeypatch)
    sys.modules.pop("mcp_server.actions", None)
    actions = importlib.import_module("mcp_server.actions")

    state = actions.EvalState(
        simulated_user_context={
            "target_category": "tray",
            "target_description": "A dark rectangular tray.",
            "source": "task_metadata",
        }
    )
    result_type, result_data, _debug = actions.dispatch_action(
        "ask",
        {
            "target_category": "tray",
            "target_description": "A guessed description.",
        },
        state,
        env=None,
    )

    assert result_type == "text"
    assert json.loads(result_data) == {
        "target_category": "tray",
        "target_description": "A dark rectangular tray.",
        "source": "task_metadata",
    }


def _import_perception_utils(monkeypatch, openai_constructor):
    omni = types.ModuleType("omni")
    replicator = types.ModuleType("omni.replicator")
    replicator_core = types.ModuleType("omni.replicator.core")
    omni.replicator = replicator
    replicator.core = replicator_core
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_core)

    openai = importlib.import_module("openai")
    monkeypatch.setattr(openai, "OpenAI", openai_constructor)
    sys.modules.pop("mcp_server.perception_utils", None)
    return importlib.import_module("mcp_server.perception_utils")


def test_perception_import_does_not_initialize_embedding_client(monkeypatch):
    calls = []

    def fail_if_initialized(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("OpenAI client must be initialized lazily")

    perception = _import_perception_utils(monkeypatch, fail_if_initialized)

    assert calls == []
    assert perception._embedding_client is None


def test_blank_optional_base_url_uses_openai_default(monkeypatch):
    calls = []
    sentinel = object()

    def capture_constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "")
    perception = _import_perception_utils(monkeypatch, capture_constructor)

    assert perception._get_embedding_client() is sentinel
    assert calls == [((), {"api_key": "test-key", "base_url": None})]


def test_exact_category_match_does_not_use_embeddings(monkeypatch):
    perception = _import_perception_utils(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail("OpenAI client was initialized"),
    )
    expected = (object(), {"1": object()})
    monkeypatch.setattr(
        perception,
        "_find_objects_exact",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        perception,
        "_get_embedding",
        lambda _text: pytest.fail("exact matching requested an embedding"),
    )

    result = perception.find_objects(
        "bread",
        {"obj_0": {"category": "bread", "original_id": "abc"}},
    )

    assert result is expected
