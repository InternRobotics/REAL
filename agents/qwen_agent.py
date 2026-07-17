"""Local-only Qwen vision-language agent for the REAL MCP server."""

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
    format_mcp_tools_for_prompt,
    run_evaluation,
)


_LEGACY_TOOL_ALIASES = {
    "nav_to": "navigate_to",
    "walk_around": "explore_receptacle",
    "gaze_at": "focus_on",
    "show_object_by_category": "find_objects",
    "show_receptacles": "highlight_receptacles",
    "place_on_top": "place",
}


@dataclass(frozen=True)
class QwenSettings:
    run: AgentRunConfig
    model_path: Path
    lora_path: Optional[Path]
    max_new_tokens: int = 4096

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "QwenSettings":
        raw_model_path = env.get("MODEL_PATH", "").strip()
        if not raw_model_path:
            raise ValueError("MODEL_PATH is required")
        model_path = Path(raw_model_path).expanduser()

        max_new_tokens = int(env.get("MAX_NEW_TOKENS", "4096"))
        if max_new_tokens <= 0:
            raise ValueError("MAX_NEW_TOKENS must be positive")

        raw_lora_path = env.get("LORA_PATH", "").strip()
        lora_path = Path(raw_lora_path).expanduser() if raw_lora_path else None
        output = env.get("EVAL_OUTPUT_PATH") or env.get("TRAJ_PATH") or "eval_output"
        run = AgentRunConfig(
            model_name=model_path.name,
            mcp_server_url=env.get("MCP_SERVER_URL", "http://127.0.0.1:8080/sse"),
            output_base=Path(output),
            max_steps=int(env.get("MAX_STEP", "30")),
            start_idx=int(env.get("START_IDX", "0")),
            end_idx=int(env.get("END_IDX", "999999")),
            result_prefix="qwen_agent",
            run_metadata={
                "lora_enabled": lora_path is not None,
            },
        )
        return cls(
            run=run,
            model_path=model_path,
            lora_path=lora_path,
            max_new_tokens=max_new_tokens,
        )


@dataclass(frozen=True)
class ParsedQwenResponse:
    summary: dict[str, Any]
    tool_name: str
    arguments: dict[str, Any]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen output does not contain a JSON object")


def parse_qwen_response(text: str) -> ParsedQwenResponse:
    """Normalize current and legacy Qwen action JSON into one decision."""

    parsed = _extract_json_object(text)
    summary = parsed.get("summary", {})
    action = parsed.get("next_action", parsed.get("next action", parsed.get("action")))
    if not isinstance(summary, dict):
        summary = {"history": str(summary)}
    if not isinstance(action, dict):
        raise ValueError("Qwen output must contain a next_action object")

    tool_name = action.get("tool_name", action.get("name", ""))
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("Qwen next_action must contain tool_name")
    tool_name = _LEGACY_TOOL_ALIASES.get(tool_name, tool_name)

    arguments = action.get("args", action.get("arguments", {}))
    if not isinstance(arguments, dict):
        raise ValueError("Qwen next_action args must be an object")
    return ParsedQwenResponse(summary, tool_name, arguments)


def build_prompt(
    task_info: Mapping[str, Any],
    mcp_tools: Sequence[Any],
    task_progress: Mapping[str, Any],
    last_action: Mapping[str, Any],
    observation_text: str,
) -> str:
    """Build one stateful action prompt from the live MCP tool schemas."""

    scene = build_scene_description(task_info.get("rooms_and_furniture", {}))
    tool_list = format_mcp_tools_for_prompt(mcp_tools)
    return f"""\
You are a vision-language robot agent operating in a home. Complete the task
using exactly one of the tools listed below per response.

Task:
{task_info["task_description"]}

Scene receptacles:
{scene or "(not provided)"}

Tools (the JSON parameter schemas are authoritative):
{tool_list}

Mandatory action preconditions:
1. The latest observation is authoritative. If it reports a failed action or
   names a prerequisite action, follow that feedback before continuing.
2. A marker_id is a short numeric label drawn in the most recent marked image.
   It is NEVER an object name such as apple_0 or a furniture name.
3. navigate_to invalidates every previous marker_id.
4. Before pick, obtain the current object marker with find_objects or
   explore_receptacle, then use the numeric label from that observation.
5. Before place, first navigate_to the destination, then call
   highlight_receptacles, and only then use the destination surface's numeric
   marker from that newest observation. Never call place immediately after
   navigate_to.
6. ask resolves semantic instruction ambiguity only. Never use ask to discover
   marker IDs or to recover from an action error.

Call finish only after the requested state has been achieved.

Previous progress summary:
{json.dumps(task_progress, ensure_ascii=False)}

Last action:
{json.dumps(last_action, ensure_ascii=False)}

Latest observation:
{observation_text or "(no observation yet)"}

Return only one JSON object with this shape:
{{
  "summary": {{
    "history": "concise factual action/observation history",
    "new_schedule": "short updated plan",
    "current_subtask": "what to do next"
  }},
  "next_action": {{"tool_name": "one listed tool", "args": {{}}}}
}}
"""


class QwenBackend:
    """Thin wrapper around a locally loaded Transformers Qwen model."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @classmethod
    def load(cls, settings: QwenSettings) -> "QwenBackend":
        if not settings.model_path.exists():
            raise SystemExit(f"[qwen_agent] Local MODEL_PATH does not exist: {settings.model_path}")
        if settings.lora_path is not None and not settings.lora_path.exists():
            raise SystemExit(f"[qwen_agent] Local LORA_PATH does not exist: {settings.lora_path}")

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except (ImportError, ModuleNotFoundError) as error:
            raise SystemExit(
                "Install a CUDA-compatible PyTorch build and then "
                "pip install -r requirements-qwen.txt"
            ) from error

        print(f"Loading local Qwen model from: {settings.model_path}")
        try:
            import accelerate  # noqa: F401

            has_accelerate = True
        except (ImportError, ModuleNotFoundError):
            has_accelerate = False

        model_kwargs = {
            "dtype": "auto",
            "local_files_only": True,
        }
        if has_accelerate:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForImageTextToText.from_pretrained(
            settings.model_path,
            **model_kwargs,
        )
        if not has_accelerate:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"accelerate is unavailable; placing the complete model on {device}.")
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            settings.model_path,
            local_files_only=True,
        )
        if settings.lora_path is not None:
            try:
                from peft import PeftModel
            except (ImportError, ModuleNotFoundError) as error:
                raise SystemExit("LORA_PATH requires peft from requirements-qwen.txt") from error
            model = PeftModel.from_pretrained(
                model,
                settings.lora_path,
                local_files_only=True,
            )
        model.eval()
        backend = cls(model, processor, settings.max_new_tokens)
        backend._torch = torch
        print("Qwen model loaded.")
        return backend

    def generate(self, prompt: str, image_paths: Sequence[Path]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": str(path)} for path in image_paths)
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise RuntimeError("Qwen returned no decoded output")
        return decoded[0]


class QwenPolicy:
    def __init__(
        self,
        backend: QwenBackend,
        task_info: Mapping[str, Any],
        mcp_tools: Sequence[Any],
    ) -> None:
        self.backend = backend
        self.task_info = task_info
        self.mcp_tools = mcp_tools
        self.task_progress: dict[str, Any] = {}
        self.last_action: dict[str, Any] = {}
        self.observation_text = ""
        self.observation_images: list[Path] = []

    async def decide(self, step_dir: Path) -> ActionDecision:
        prompt = build_prompt(
            self.task_info,
            self.mcp_tools,
            self.task_progress,
            self.last_action,
            self.observation_text,
        )
        (step_dir / "user_message.txt").write_text(prompt, encoding="utf-8")
        raw_response = self.backend.generate(prompt, self.observation_images)
        parsed = parse_qwen_response(raw_response)
        self.task_progress = parsed.summary
        return ActionDecision(
            tool_name=parsed.tool_name,
            arguments=parsed.arguments,
            raw_response=raw_response,
        )

    def observe(self, decision: ActionDecision, observation: ToolObservation) -> None:
        self.last_action = {
            "tool_name": decision.tool_name,
            "args": decision.arguments,
        }
        observation_parts = []
        if observation.result_type == "text" and observation.data:
            observation_parts.append(f"Text: {observation.data}")
        if observation.image_paths:
            observation_parts.append(
                f"Images: {' '.join('<image>' for _ in observation.image_paths)}"
            )
        self.observation_text = "\n".join(observation_parts) or "No user-facing content returned."
        self.observation_images = observation.image_paths[-2:]


async def main_async(settings: QwenSettings) -> None:
    backend = QwenBackend.load(settings)
    print(f"\nQwen agent: {settings.run.model_name}")
    print(f"MCP server: {settings.run.mcp_server_url}")
    print(f"Episode range: [{settings.run.start_idx}, {settings.run.end_idx})")
    print(f"Output: {settings.run.output_base}")
    print(f"Max steps per episode: {settings.run.max_steps}\n")

    def make_policy(task_info: dict[str, Any], tools: Sequence[Any], _path: Path) -> QwenPolicy:
        return QwenPolicy(backend, task_info, tools)

    await run_evaluation(settings.run, make_policy)


def main() -> None:
    try:
        settings = QwenSettings.from_env()
    except (TypeError, ValueError) as error:
        raise SystemExit(f"[qwen_agent] Invalid configuration: {error}") from error
    asyncio.run(main_async(settings))


if __name__ == "__main__":
    main()
