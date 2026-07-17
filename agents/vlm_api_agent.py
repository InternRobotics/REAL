"""OpenAI-compatible VLM API agent for the REAL MCP server."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agents.common import (
    ActionDecision,
    AgentRunConfig,
    ToolObservation,
    build_scene_description,
    mcp_tool_to_openai,
    run_evaluation,
)


SYSTEM_PROMPT = """\
You are an embodied agent operating in a home. Complete the household task by
using the available tools and the RGB observations returned by the simulator.

# Dynamic marker IDs
For `focus_on`, `pick`, `place`, `open`, and `close`, use only a marker ID from
the most recent observation. Marker mappings change after actions; never reuse
an old marker ID without checking the latest observation.

# Interaction
Use `ask` when the instruction is ambiguous and visual evidence gives you a
candidate category and description to verify.

# Completion
Call `finish` as soon as the requested world state has been achieved. Do not
ask the user for confirmation before finishing.

Call exactly one tool per turn.\
"""


@dataclass(frozen=True)
class VlmApiSettings:
    run: AgentRunConfig
    api_key: Optional[str]
    base_url: Optional[str]
    max_tokens: int = 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "VlmApiSettings":
        model_name = env.get("MODEL_NAME", "").strip()
        if not model_name:
            raise ValueError("MODEL_NAME is required")

        max_tokens = int(env.get("MAX_NEW_TOKENS", "1024"))
        if max_tokens <= 0:
            raise ValueError("MAX_NEW_TOKENS must be positive")

        output = env.get("EVAL_OUTPUT_PATH") or env.get("TRAJ_PATH") or "eval_output"
        run = AgentRunConfig(
            model_name=model_name,
            mcp_server_url=env.get("MCP_SERVER_URL", "http://127.0.0.1:8080/sse"),
            output_base=Path(output),
            max_steps=int(env.get("MAX_STEP", "20")),
            start_idx=int(env.get("START_IDX", "0")),
            end_idx=int(env.get("END_IDX", "999999")),
            result_prefix="vlm_api_agent",
        )
        base_url = env.get("OPENAI_API_BASE_URL") or env.get("OPENAI_BASE_URL") or None
        return cls(
            run=run,
            api_key=env.get("OPENAI_API_KEY") or None,
            base_url=base_url,
            max_tokens=max_tokens,
        )


class VlmApiPolicy:
    """Function-calling policy backed by an OpenAI-compatible chat API."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        task_info: Mapping[str, Any],
        mcp_tools: Sequence[Any],
        max_tokens: int,
    ) -> None:
        scene = build_scene_description(task_info.get("rooms_and_furniture", {}))
        system_prompt = SYSTEM_PROMPT
        if scene:
            system_prompt += (
                "\n\n# Scene\nThe scene contains these rooms and receptacles:\n" + scene
            )
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.openai_tools = [mcp_tool_to_openai(tool) for tool in mcp_tools]
        self.messages: list[Any] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Your task is: {task_info['task_description']}",
            },
        ]
        self.current_images: list[str] = []

    async def decide(self, _step_dir: Path) -> ActionDecision:
        messages = list(self.messages)
        if self.current_images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        }
                        for image in self.current_images
                    ],
                }
            )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.openai_tools,
            tool_choice="auto",
            max_tokens=self.max_tokens,
        )
        if not response.choices:
            raise RuntimeError("VLM API returned no choices")

        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        raw_response = {
            "role": message.role,
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": (
                            json.dumps(tool_call.function.arguments)
                            if isinstance(tool_call.function.arguments, dict)
                            else tool_call.function.arguments
                        ),
                    },
                }
                for tool_call in tool_calls
            ],
        }
        history_message = {
            "role": message.role,
            "content": message.content,
        }
        if tool_calls:
            # The runner intentionally executes one action per turn. Keeping
            # unexecuted calls in history would leave unresolved tool_call IDs
            # and make strict OpenAI-compatible endpoints reject the next turn.
            history_message["tool_calls"] = raw_response["tool_calls"][:1]
        self.messages.append(history_message)

        if not tool_calls:
            self.messages.append(
                {
                    "role": "user",
                    "content": "Select exactly one available tool to continue the task.",
                }
            )
            return ActionDecision(None, {}, raw_response)

        tool_call = tool_calls[0]
        raw_arguments = tool_call.function.arguments
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        return ActionDecision(
            tool_name=tool_call.function.name,
            arguments=arguments,
            raw_response=raw_response,
            tool_call_id=tool_call.id,
        )

    def observe(self, decision: ActionDecision, observation: ToolObservation) -> None:
        content = (
            observation.data
            if observation.result_type == "text"
            else "[image observation attached to the next model input]"
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": decision.tool_call_id,
                "content": content or "[empty tool response]",
            }
        )
        if observation.images_b64:
            self.current_images.extend(observation.images_b64[-1:])
            self.current_images = self.current_images[-2:]


async def main_async(settings: VlmApiSettings) -> None:
    try:
        import openai
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Install runtime dependencies with: pip install -r requirements.txt"
        ) from error

    if not settings.api_key and settings.base_url is None:
        print("[vlm_api_agent] OPENAI_API_KEY is unset; the default API endpoint may reject calls.")

    client = openai.AsyncOpenAI(
        api_key=settings.api_key or "not-needed",
        base_url=settings.base_url,
    )
    print(f"\nVLM API agent: {settings.run.model_name}")
    print(f"MCP server: {settings.run.mcp_server_url}")
    print(f"Episode range: [{settings.run.start_idx}, {settings.run.end_idx})")
    print(f"Output: {settings.run.output_base}")
    print(f"Max steps per episode: {settings.run.max_steps}\n")

    def make_policy(task_info: dict[str, Any], tools: Sequence[Any], _path: Path) -> VlmApiPolicy:
        return VlmApiPolicy(
            client=client,
            model_name=settings.run.model_name,
            task_info=task_info,
            mcp_tools=tools,
            max_tokens=settings.max_tokens,
        )

    await run_evaluation(settings.run, make_policy)


def main() -> None:
    try:
        settings = VlmApiSettings.from_env()
    except (TypeError, ValueError) as error:
        raise SystemExit(f"[vlm_api_agent] Invalid configuration: {error}") from error
    asyncio.run(main_async(settings))


if __name__ == "__main__":
    main()
