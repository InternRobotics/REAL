"""Network-level closed-loop tests for both agent backends over MCP SSE."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.common import AgentRunConfig, run_evaluation
from agents.qwen_agent import QwenPolicy
from agents.vlm_api_agent import VlmApiPolicy
from tests.mcp_validation_server import TARGET_WORLD_GRAPH


ACTION_PLAN = [
    ("navigate_to", {"receptacle_name": "source_counter"}),
    ("find_objects", {"target_category": "apple"}),
    ("pick", {"marker_id": "0"}),
    ("navigate_to", {"receptacle_name": "target_table"}),
    ("highlight_receptacles", {}),
    ("place", {"marker_id": "0"}),
    ("finish", {}),
]


class _ValidationServer:
    def __init__(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "mcp_calls.json"
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/sse"

    def start(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(self.port),
                "VALIDATION_LOG_PATH": str(self.log_path),
            }
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "tests.mcp_validation_server"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"validation MCP server exited with {self.process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                pass
            time.sleep(0.02)
        raise RuntimeError("validation MCP server did not start")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.temp_dir.cleanup()

    @property
    def state(self):
        payload = json.loads(self.log_path.read_text(encoding="utf-8"))
        return SimpleNamespace(
            completed=payload["completed"],
            world_graph=payload["world_graph"],
            calls=payload["calls"],
        )


class _QueuedQwenBackend:
    def __init__(self) -> None:
        self.actions = list(ACTION_PLAN)
        self.prompts: list[str] = []

    def generate(self, prompt: str, image_paths: list[Path]) -> str:
        self.prompts.append(prompt)
        tool_name, arguments = self.actions.pop(0)
        return json.dumps(
            {
                "summary": {
                    "history": f"selected {tool_name}",
                    "new_schedule": "continue the pick-and-place task",
                    "current_subtask": tool_name,
                },
                "next_action": {"tool_name": tool_name, "args": arguments},
            }
        )


class _QueuedCompletions:
    def __init__(self) -> None:
        self.actions = list(ACTION_PLAN)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        tool_name, arguments = self.actions.pop(0)
        tool_call = SimpleNamespace(
            id=f"call-{len(self.requests)}",
            function=SimpleNamespace(
                name=tool_name,
                arguments=json.dumps(arguments),
            ),
        )
        message = SimpleNamespace(role="assistant", content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentMcpClosedLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.validation_server = _ValidationServer()
        self.validation_server.start()
        self.output_tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.validation_server.stop()
        self.output_tmp.cleanup()

    def _config(self, model_name: str, result_prefix: str) -> AgentRunConfig:
        return AgentRunConfig(
            model_name=model_name,
            mcp_server_url=self.validation_server.url,
            output_base=Path(self.output_tmp.name),
            max_steps=10,
            result_prefix=result_prefix,
        )

    def _assert_closed_loop(self, summary: dict[str, Any], model_slug: str) -> None:
        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["finished"], 1)
        self.assertEqual(summary["successes"], 1)
        self.assertNotIn("mcp_server_url", summary)
        self.assertEqual(self.validation_server.state.world_graph, TARGET_WORLD_GRAPH)
        self.assertTrue(self.validation_server.state.completed)

        calls = self.validation_server.state.calls
        self.assertEqual(calls[0]["phase"], "load_task")
        self.assertEqual(calls[-1]["phase"], "complete_task")
        self.assertEqual(
            [call["tool_name"] for call in calls[1:-1]],
            [tool_name for tool_name, _arguments in ACTION_PLAN[:-1]],
        )

        trajectory = (
            Path(self.output_tmp.name)
            / "mcp_validation_scene"
            / "mcp-closed-loop-pick-place"
            / model_slug
        )
        episode_summary = json.loads(
            (trajectory / "episode_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(episode_summary["success"])
        self.assertEqual(episode_summary["status"], "finished")
        self.assertTrue((trajectory / "step_000" / "image_observation_0.png").is_file())
        self.assertTrue((trajectory / "step_005" / "debug_info.json").is_file())

    async def test_vlm_api_policy_completes_real_mcp_sse_loop(self):
        completions = _QueuedCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = self._config("vlm-e2e", "vlm_api_agent")

        def make_policy(task_info, tools, _trajectory):
            return VlmApiPolicy(client, config.model_name, task_info, tools, max_tokens=128)

        summary = await run_evaluation(config, make_policy)

        self._assert_closed_loop(summary, "vlm_e2e")
        self.assertEqual(len(completions.requests), len(ACTION_PLAN))
        image_messages = [
            content
            for request in completions.requests[1:]
            for message in request["messages"]
            if isinstance(message.get("content"), list)
            for content in message["content"]
            if content.get("type") == "image_url"
        ]
        self.assertTrue(image_messages)

    async def test_qwen_policy_completes_real_mcp_sse_loop(self):
        backend = _QueuedQwenBackend()
        config = self._config("qwen-e2e", "qwen_agent")

        def make_policy(task_info, tools, _trajectory):
            return QwenPolicy(backend, task_info, tools)

        summary = await run_evaluation(config, make_policy)

        self._assert_closed_loop(summary, "qwen_e2e")
        self.assertEqual(len(backend.prompts), len(ACTION_PLAN))
        self.assertIn('"name": "finish"', backend.prompts[0])
        self.assertIn("Never call place immediately after", backend.prompts[0])
        self.assertIn("Images: <image>", backend.prompts[1])


if __name__ == "__main__":
    unittest.main()
