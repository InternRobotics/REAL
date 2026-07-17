"""Regression tests for the two MCP agent clients."""

from __future__ import annotations

import base64
import importlib
import json
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image


class AgentHelperTests(unittest.TestCase):
    def test_agent_modules_are_import_safe_without_model_or_api_configuration(self):
        importlib.import_module("agents.vlm_api_agent")
        importlib.import_module("agents.qwen_agent")

    def test_mcp_tools_drive_both_backend_formats(self):
        from agents.common import format_mcp_tools_for_prompt, mcp_tool_to_openai
        from mcp_server.tools import MCP_TOOLS

        openai_tools = [mcp_tool_to_openai(tool) for tool in MCP_TOOLS]
        prompt_tools = format_mcp_tools_for_prompt(MCP_TOOLS)

        names = [tool["function"]["name"] for tool in openai_tools]
        self.assertIn("navigate_to", names)
        self.assertIn("finish", names)
        self.assertNotIn("nav_to", names)
        self.assertIn('"target_description"', prompt_tools)
        self.assertNotIn('"question"', prompt_tools)

    def test_qwen_response_parser_accepts_think_and_code_fence(self):
        from agents.qwen_agent import parse_qwen_response

        response = parse_qwen_response(
            "<think>private reasoning</think>\n"
            "```json\n"
            '{"summary":{"history":"looked"},'
            '"next_action":{"tool_name":"finish","args":{}}}\n'
            "```"
        )

        self.assertEqual(response.tool_name, "finish")
        self.assertEqual(response.arguments, {})
        self.assertEqual(response.summary, {"history": "looked"})

    def test_qwen_response_parser_normalizes_source_agent_tool_names(self):
        from agents.qwen_agent import parse_qwen_response

        response = parse_qwen_response(
            '{"next action":{"tool_name":"nav_to","args":{"receptacle_name":"table"}}}'
        )

        self.assertEqual(response.tool_name, "navigate_to")

    def test_parse_tool_result_extracts_debug_graph_and_image(self):
        from agents.common import extract_world_graph_from_debug, parse_tool_result

        image = base64.b64encode(b"not-decoded-in-this-test").decode("ascii")
        result = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="picked"),
                SimpleNamespace(type="image", data=image),
                SimpleNamespace(
                    type="text",
                    text='Debug Info:\n{"world_graph":{"table":{"content":["apple"]}}}',
                ),
            ]
        )

        observation = parse_tool_result(result)

        self.assertEqual(observation.result_type, "text")
        self.assertEqual(observation.data, "picked")
        self.assertEqual(observation.images_b64, [image])
        self.assertEqual(
            extract_world_graph_from_debug(observation.debug_info),
            {"table": {"content": ["apple"]}},
        )

    def test_world_graph_evaluation_checks_articulation_state(self):
        from agents.common import evaluate_world_graph

        target = {"cabinet": {"content": ["apple"], "door": False}}

        self.assertTrue(
            evaluate_world_graph(
                {"cabinet": {"content": ["apple"], "door": False}},
                target,
            )
        )
        self.assertFalse(
            evaluate_world_graph(
                {"cabinet": {"content": ["apple"], "door": True}},
                target,
            )
        )
        self.assertFalse(evaluate_world_graph({"cabinet": {"content": []}}, {}))

    def test_episode_output_uses_globally_unique_benchmark_id(self):
        from agents.common import AgentRunConfig, episode_output_dir

        config = AgentRunConfig(
            model_name="test-model",
            mcp_server_url="http://127.0.0.1:8080/sse",
        )
        first = episode_output_dir(
            config,
            {
                "scene_id": "scene",
                "task_id": "task_1",
                "benchmark_task_id": "FDP/task_1",
            },
        )
        second = episode_output_dir(
            config,
            {
                "scene_id": "scene",
                "task_id": "task_1",
                "benchmark_task_id": "SUL/task_1",
            },
        )

        self.assertNotEqual(first, second)

    def test_qwen_loader_forces_local_only_model_and_processor(self):
        from agents.common import AgentRunConfig
        from agents.qwen_agent import QwenBackend, QwenSettings

        calls = {}

        class FakeModel:
            def eval(self):
                calls["eval"] = True

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["model"] = (path, kwargs)
                return FakeModel()

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["processor"] = (path, kwargs)
                return object()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForImageTextToText = FakeAutoModel
        fake_transformers.AutoProcessor = FakeAutoProcessor
        fake_torch = types.ModuleType("torch")
        fake_accelerate = types.ModuleType("accelerate")

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = Path(model_dir)
            settings = QwenSettings(
                run=AgentRunConfig(
                    model_name="qwen",
                    mcp_server_url="http://127.0.0.1:8080/sse",
                ),
                model_path=model_path,
                lora_path=None,
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "accelerate": fake_accelerate,
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                },
            ):
                QwenBackend.load(settings)

        self.assertTrue(calls["model"][1]["local_files_only"])
        self.assertTrue(calls["processor"][1]["local_files_only"])
        self.assertTrue(calls["eval"])

    def test_qwen_run_metadata_does_not_record_local_paths(self):
        from agents.qwen_agent import QwenSettings

        with tempfile.TemporaryDirectory() as model_dir:
            settings = QwenSettings.from_env(
                {
                    "MODEL_PATH": model_dir,
                    "LORA_PATH": str(Path(model_dir) / "adapter"),
                }
            )

        self.assertNotIn("model_path", settings.run.run_metadata)
        self.assertNotIn("lora_path", settings.run.run_metadata)
        self.assertTrue(settings.run.run_metadata["lora_enabled"])


class EpisodeRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_vlm_policy_executes_only_first_returned_tool_call(self):
        from agents.common import ToolObservation
        from agents.vlm_api_agent import VlmApiPolicy
        from mcp_server.tools import MCP_TOOLS

        tool_calls = [
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="navigate_to",
                    arguments='{"receptacle_name":"table"}',
                ),
            ),
            SimpleNamespace(
                id="call-2",
                function=SimpleNamespace(name="finish", arguments="{}"),
            ),
        ]
        message = SimpleNamespace(role="assistant", content=None, tool_calls=tool_calls)

        class Completions:
            async def create(self, **_kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        policy = VlmApiPolicy(
            client,
            "test-model",
            {
                "task_description": "go to the table",
                "rooms_and_furniture": {"room": ["table"]},
            },
            MCP_TOOLS,
            32,
        )

        decision = await policy.decide(Path("unused"))
        policy.observe(decision, ToolObservation("text", "arrived", {}, []))

        self.assertEqual(decision.tool_name, "navigate_to")
        assistant_history = policy.messages[-2]
        self.assertEqual(len(assistant_history["tool_calls"]), 1)
        self.assertEqual(assistant_history["tool_calls"][0]["id"], "call-1")

    async def test_episode_executes_action_then_scores_finish(self):
        from agents.common import ActionDecision, AgentRunConfig, run_episode

        target_graph = {"table": {"content": ["apple"]}}
        debug = json.dumps({"world_graph": target_graph})
        image_buffer = BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

        class FakeSession:
            def __init__(self):
                self.calls = []

            async def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                if name == "navigate_to":
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="arrived"),
                            SimpleNamespace(type="image", data=image_b64),
                            SimpleNamespace(type="text", text=f"Debug Info:\n{debug}"),
                        ]
                    )
                if name == "finish":
                    return SimpleNamespace(
                        content=[
                            SimpleNamespace(type="text", text="All evaluation episodes completed.")
                        ]
                    )
                raise AssertionError(f"unexpected tool: {name}")

        class FakePolicy:
            def __init__(self):
                self.index = 0
                self.observations = []

            async def decide(self, _step_dir):
                decisions = [
                    ActionDecision(
                        tool_name="navigate_to",
                        arguments={"receptacle_name": "table"},
                        raw_response={"step": 0},
                    ),
                    ActionDecision(
                        tool_name="finish",
                        arguments={},
                        raw_response={"step": 1},
                    ),
                ]
                decision = decisions[self.index]
                self.index += 1
                return decision

            def observe(self, decision, observation):
                self.observations.append((decision, observation))

        task = {
            "task_id": "task-1",
            "episode_idx": 0,
            "scene_id": "scene-1",
            "task_description": "put the apple on the table",
            "initial_world_graph": {"table": {"content": []}},
            "target_world_graph": target_graph,
        }

        with tempfile.TemporaryDirectory() as output_dir:
            config = AgentRunConfig(
                model_name="test-model",
                mcp_server_url="http://127.0.0.1:8080/sse",
                output_base=Path(output_dir),
                max_steps=3,
            )
            session = FakeSession()
            policy = FakePolicy()

            summary, next_task = await run_episode(session, task, config, policy)

            self.assertTrue(summary["success"])
            self.assertEqual(summary["status"], "finished")
            self.assertIsNone(next_task)
            self.assertEqual(
                session.calls,
                [
                    ("navigate_to", {"receptacle_name": "table"}),
                    ("finish", {}),
                ],
            )
            self.assertEqual(len(policy.observations), 1)
            self.assertEqual(len(policy.observations[0][1].image_paths), 1)
            self.assertTrue(policy.observations[0][1].image_paths[0].is_file())
            episode_dir = Path(output_dir) / "scene-1" / "task-1" / "test_model"
            self.assertTrue((episode_dir / "episode_summary.json").is_file())
            self.assertTrue((episode_dir / "step_000" / "tool_call.json").is_file())
            self.assertTrue((episode_dir / "step_001" / "tool_call.json").is_file())


if __name__ == "__main__":
    unittest.main()
