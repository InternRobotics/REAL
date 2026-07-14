"""
annotate_trajectory.py — Annotate embodied manipulation trajectories with Chain-of-Thought reasoning.

Reads PKL trajectory files and their meta JSON files, calls an OpenAI-compatible
LLM to generate CoT annotations for each step, and writes annotated JSON output.

Usage:
    python annotate_trajectory.py --config config_example.json
"""

import argparse
import base64
import ast
import copy
import json
import os
import pickle as pkl
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image as PILImage

from pkl_data_reader import _normalize_action_dict

# ------------ Config ------------
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEBUG_IMAGE = os.getenv("ANNO_DEBUG_IMAGE", "1").strip() in ("1", "true", "True", "YES", "yes")
REQUEST_TIMEOUT_S = float(os.getenv("ANNO_REQUEST_TIMEOUT_S", "90"))
EPISODE_WORKERS = int(os.getenv("ANNO_EPISODE_WORKERS", "100"))

if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY.")

def make_openai_client() -> OpenAI:
    """Create a fresh OpenAI client (one per episode worker)."""
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_S)

# ------------ Jobs Config ------------
# Jobs are loaded from an external JSON config file via --config.
# See config_example.json for the expected format.

# ------------ PKL Processing Utils ------------

def _to_uint8_rgb(arr):
    if arr is None:
        return None
    import numpy as _np
    if arr.ndim == 2:
        arr = _np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    y = arr.astype(_np.float32)
    if _np.nanmax(y) <= 1.0:
        y *= 255.0
    y = _np.clip(y, 0, 255).astype(_np.uint8)
    return y[:, :, :3]


def iter_paired_steps(traj):
    """Yield (action, obs) pairs from trajectory."""
    pending_action = None
    for item in traj[1:]:
        if not isinstance(item, dict) or "action" not in item:
            continue
        act = item.get("action")
        payload = item.get("payload")
        if act == "error":
            continue
        if act == "obs":
            yield pending_action, {"payload": payload}
            pending_action = None
            continue
        if pending_action is None:
            pending_action = {"name": act, "payload": payload}
        else:
            yield pending_action, None
            pending_action = {"name": act, "payload": payload}
    if pending_action is not None:
        yield pending_action, None


# ------------ Annotation Utils (from anno_batch.py) ------------
def image_path_to_data_url(image_path: str) -> str:
    p = Path(image_path)
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    # All images are confirmed to be PNG.
    return f"data:image/png;base64,{b64}"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_template_record(tpl_path: Path) -> Dict[str, Any]:
    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))
    if isinstance(tpl, list):
        for rec in tpl:
            if isinstance(rec, dict) and rec:
                return rec
    if isinstance(tpl, dict):
        return tpl
    raise ValueError(f"No usable template record in {tpl_path}")


def find_system_prompt(sample: Dict[str, Any]) -> str:
    convs = sample.get("conversations", [])
    for c in convs:
        if c.get("from") == "system" and str(c.get("value", "")).strip():
            return c["value"]
    return convs[0].get("value", "") if convs and convs[0].get("from") == "system" else ""


_SPECIAL_TOKEN_RE = re.compile(r"</?\|[^>]+\|>")
_TOOL_INVOCATION_RE = re.compile(r"<tool_invocation[\s\S]*?/>")


def strip_special_tokens(text: str) -> str:
    if not text:
        return ""
    cleaned = _SPECIAL_TOKEN_RE.sub("", text)
    cleaned = _TOOL_INVOCATION_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return ""
    in_str = False
    esc = False
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return candidate
            continue
    return s[start:]


def fix_json_newlines_in_strings(s: str) -> str:
    """Fix illegal newlines inside JSON string values.
    
    JSON standard doesn't allow literal newlines inside string values.
    This function replaces them with escaped \\n sequences.
    """
    result = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                result.append(ch)
                esc = False
                continue
            if ch == '\\':
                result.append(ch)
                esc = True
                continue
            if ch == '"':
                result.append(ch)
                in_str = False
                continue
            # Replace literal newlines with escaped version
            if ch == '\n':
                result.append('\\n')
                continue
            if ch == '\r':
                result.append('\\r')
                continue
            result.append(ch)
        else:
            if ch == '"':
                in_str = True
            result.append(ch)
    return ''.join(result)


def parse_json_or_none(s: Any) -> Optional[Any]:
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    # First try direct parse
    try:
        return json.loads(t)
    except Exception:
        pass
    # Try fixing newlines in string values
    try:
        fixed = fix_json_newlines_in_strings(t)
        return json.loads(fixed)
    except Exception:
        return None


def _coerce_new_schedule(v: Any) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, list):
        out = [str(x).strip() for x in v if x is not None and str(x).strip()]
        return out or None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        parts = re.split(r"(?:\n+|(?:\d+\.\s+)|(?:\d+\)\s+)|(?:;\s+))", s)
        out = [p.strip(" \t-") for p in parts if p and p.strip(" \t-")]
        return out or [s]
    return None


def canonicalize_thinking_dict(obj: Dict[str, Any], include_last_action: bool = False) -> Dict[str, Any]:
    """Canonicalize thinking dict with optional Last Action field.
    
    Args:
        obj: Raw thinking dict from GPT
        include_last_action: If True, preserve "Last Action" field in Summary
    """
    out: Dict[str, Any] = {"CoT": str(obj.get("CoT", "") or "")}
    s_in = obj.get("Summary")
    s = s_in if isinstance(s_in, dict) else {}
    
    summary_out = {
        "History": str(s.get("History", "") or ""),
        "New Schedule": _coerce_new_schedule(s.get("New Schedule")),
        "Current subtask": str(s.get("Current subtask", "") or ""),
    }
    
    # Preserve Last Action if requested (for step >= 1 input)
    if include_last_action and "Last Action" in s:
        summary_out["Last Action"] = s.get("Last Action")
    
    out["Summary"] = summary_out
    return out


def call_chat_thinking(
    client: OpenAI,
    system_prompt: str,
    human_prompt: str,
    image_path: Optional[str],
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """Call GPT with retry logic."""
    for attempt in range(max_retries):
        try:
            if image_path and Path(image_path).exists():
                # Check image size
                img_size_mb = Path(image_path).stat().st_size / (1024 * 1024)
                if img_size_mb > 10:
                    print(f"[WARNING] Large image: {img_size_mb:.2f}MB, may cause timeout")
                
                user_content = [
                    {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}},
                    {"type": "text", "text": human_prompt},
                ]
            else:
                user_content = human_prompt
            
            # Try with response_format for strict JSON output (may not be supported by all models)
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Fallback without response_format if not supported
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                )
            return (resp.choices[0].message.content or "").strip()
        
        except Exception as e:
            print(f"[ERROR] GPT call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}")
            if attempt == max_retries - 1:
                # Last attempt failed, try text-only fallback
                try:
                    print(f"[FALLBACK] Trying text-only (no image)")
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": human_prompt},
                        ],
                        temperature=temperature,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e2:
                    print(f"[ERROR] Text-only fallback also failed: {str(e2)[:200]}")
                    return ""  # Return empty string if all retries failed
            # Wait before retry
            import time
            time.sleep(2)
    
    # Should never reach here, but just in case
    return ""


def build_human_prompt_step0(
    instruction: str, init_action: str, object_info: Optional[Dict[str, Any]] = None
) -> str:

    # human prompt from v3
    input_obj = {
        "Task instruction": instruction or "",
        "History": {},
        "Observation": {
            "camera": None,  # No image for step 0
            "text": "",
            "conversation": []
        },
        "Next Action": init_action,
        "IMPORTANT": "You are annotating Chain-of-Thought reasoning for a SUCCESSFUL execution trajectory. The Next Action shown is from ground truth and will be executed correctly. Your task is to write the CoT that explains WHY this action is reasonable given the current state. You MUST output ONLY a single valid JSON object. Do NOT output any other text or explanations.\n",

        "Object disambiguation metadata": "1.src: Task-required source Object\n 2.src_distractors: Source distractors (do NOT treat as src)\n 3.dest: Task-required destination Object \n 4.dest_distractors: Destination distractors (do NOT treat as dest)\n",
        "Object disambiguation": object_info or {},

        "Output format (REQUIRED JSON)": {
            "CoT": "",
            "Summary": {
                "History": "0) I have already received the detailed task instruction.",
                "New Schedule": "",
                "Current subtask": "",
            },
        
        },
    }
    return json.dumps(input_obj, ensure_ascii=False, indent=2)


def build_human_prompt_stepN(
    *,
    instruction: str,
    prev_action_text: str,
    next_acton: str,
    accumulated_history: str,
    obs_dict: Dict[str, Any],
    object_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Build step >= 1 prompt with accumulated History.
    
    Args:
        prev_action_text: JSON string of last executed action
        next_acton: JSON string of next action to execute
        accumulated_history: Script-maintained accumulated history string (all past steps)
    """
    # Build history object with accumulated history
    last_action_obj = parse_json_or_none(prev_action_text) if prev_action_text else {}
    history_obj: Dict[str, Any] = {
        "summary": {
            "History": accumulated_history,
            "Last Action": last_action_obj if last_action_obj else {},
        }
    }

    # Build Observation object (camera + conversation only)
    if obs_dict['text'].startswith('.....') or obs_dict['text'].startswith('.....'):
        conversation = [obs_dict['text']]
        obs_text = ''
    else:
        obs_text = obs_dict['text']
        conversation = []
        
    obs_obj = {
        "camera": "<image>",
        "text": obs_dict['text'],
        "conversation": [],
    }
    
    input_obj = {
        "Task instruction": instruction or "",
        "History": history_obj,
        "Observation": obs_obj,
        "Next Action": next_acton,
        "IMPORTANT": "You MUST output ONLY a single valid JSON object. Do NOT output any other text. The Last Action in History.summary has been SUCCESSFULLY EXECUTED and the Observation shows the result.Your COT and plan MUST correspond to the next action you will take.",
        "Object disambiguation metadata": "1.src: Task-required source Object\n 2.src_distractors: Source distractors (do NOT treat as src)\n 3.dest: Task-required destination Object \n 4.dest_distractors: Destination distractors (do NOT treat as dest)\n",
        "Object disambiguation": object_info or {},
        "Output format (REQUIRED JSON)": {
            "CoT": "your reasoning as a string",
            "Summary": {
                "History": "ONLY describe the result of the Last Action just executed. Do NOT repeat previous history.",
                "New Schedule": "",
                "Current subtask": "",
                "Last Action": "the previous action as a dict (e.g. {'tool_name': '...', 'args': {...}}) - this action has been SUCCESSFULLY EXECUTED",
            },
        },
    }
    return json.dumps(input_obj, ensure_ascii=False, indent=2)


def format_action_dict(action_dict: Dict[str, Any]) -> str:
    """Format action dict to canonical JSON string."""
    if not action_dict:
        return ""
    
    def _maybe_parse_py_dict_str(s: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(s, str):
            return None
        t = s.strip()
        if not (t.startswith("{") and t.endswith("}")):
            return None
        try:
            obj = ast.literal_eval(t)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _normalize_tool_and_args(tool_name: str, raw_args: Any) -> Tuple[str, Dict[str, Any]]:
        # Tool name normalization
        if tool_name == "walkaround":
            tool_name = "walk_around"
        if tool_name == "place":
            tool_name = "place"
        if tool_name == "show_object_by_category":
            tool_name = "show_object_by_category"
        if tool_name in {"walk_around", "show_receptacles", "finish","check_scene_objects"}:
            return tool_name, {}
        
        # Tool -> required parameter name mapping
        tool_param_map = {
            "gaze_at": "marker_id",
            "show_object_by_category": "target_category",
            "nav_to": "receptacle_name",
            "pick": "marker_id",
            "place": "marker_id",
            "open": "receptacle_name",
            "close": "receptacle_name",
            "ask": "question",
        }
        
        args_obj: Any = raw_args
        if args_obj in ("", None):
            args_obj = {}
        if isinstance(args_obj, str):
            parsed = _maybe_parse_py_dict_str(args_obj)
            if parsed is not None:
                args_obj = parsed
        if isinstance(args_obj, dict):
            for k, v in list(args_obj.items()):
                parsed = _maybe_parse_py_dict_str(v)
                if parsed is not None:
                    if k in parsed and len(parsed) == 1:
                        args_obj[k] = parsed[k]
                    else:
                        args_obj[k] = parsed
        if not isinstance(args_obj, dict):
            param_name = tool_param_map.get(tool_name, "value")
            args_obj = {param_name: args_obj}
        return tool_name, args_obj

    if isinstance(action_dict, dict) and len(action_dict) == 1:
        tool_name = next(iter(action_dict.keys()))
        raw_args = action_dict[tool_name]
        tool_name2, args2 = _normalize_tool_and_args(tool_name, raw_args)
        tool_call = {"tool_name": tool_name2, "args": args2}
        try:
            return json.dumps(tool_call, ensure_ascii=False)
        except Exception:
            return str(tool_call)
    try:
        return json.dumps(action_dict, ensure_ascii=False)
    except Exception:
        return str(action_dict)


def set_human_value(sample: Dict[str, Any], value: str) -> None:
    for c in sample.get("conversations", []):
        if c.get("from") == "human":
            c["value"] = value
            return
    sample.setdefault("conversations", []).append({"from": "human", "value": value})


def set_gpt_thinking(sample: Dict[str, Any], thinking: str) -> None:
    for c in sample.get("conversations", []):
        if c.get("from") == "gpt" and c.get("think_or_action", 0) == 0:
            c["value"] = thinking
            c["think_or_action"] = 0
            c["end_turn"] = False
            return
    sample.setdefault("conversations", []).append(
        {"from": "gpt", "value": thinking, "think_or_action": 0, "end_turn": False}
    )


def set_gpt_action(sample: Dict[str, Any], action_text: str) -> None:
    for c in sample.get("conversations", []):
        if c.get("from") == "gpt" and c.get("think_or_action", 1) == 1:
            c["value"] = action_text
            c["think_or_action"] = 1
            c["end_turn"] = True
            return
    sample.setdefault("conversations", []).append(
        {"from": "gpt", "value": action_text, "think_or_action": 1, "end_turn": True}
    )


@dataclass
class Step:
    episode_id: str
    step_id: int
    sample: Dict[str, Any]
    action_dict: Dict[str, Any]  # From PKL
    obs_text: str  # From PKL
    image_path: Optional[str]  # Saved image path


def process_pkl_to_steps(
    pkl_path: Path,
    episode_id: str,
    output_dir: Path,
    base_rec: Dict[str, Any],
    episode_meta: Optional[Dict[str, Any]] = None,
) -> List[Step]:
    """Extract steps from a single PKL file."""
    traj_obj = pkl.load(pkl_path.open("rb"))
    ep_img_dir = output_dir / "obs" / "images" / episode_id
    ep_img_dir.mkdir(parents=True, exist_ok=True)

    steps: List[Step] = []
    step_id = 1  # Start from 1 (step 0 reserved for virtual instruction-only step)
    
    for action, obs in iter_paired_steps(traj_obj):
        # Extract action dict
        action_dict = {}
        if action:
            act_dict = {"action": action["name"], "payload": action["payload"]}
            norm = _normalize_action_dict(act_dict) or {action["name"]: action["payload"]}
            action_dict = norm

        # Extract observation
        obs_payload = obs.get("payload") if obs else None
        obs_text_str = ""
        if isinstance(obs_payload, str) and obs_payload.strip():
            obs_text_str = obs_payload.strip()

        # Save image
        image_path_abs = None
        image_path_rel = ""
        img_obj = None
        if isinstance(obs_payload, PILImage.Image):
            img_obj = obs_payload
        elif hasattr(obs_payload, "shape"):
            arr = _to_uint8_rgb(obs_payload)
            if arr is not None:
                img_obj = PILImage.fromarray(arr)
        if img_obj is not None:
            img_path = ep_img_dir / f"{episode_id}_step_{step_id}.png"
            img_obj.save(img_path)
            image_path_abs = str(img_path)
            # Store relative path for portability
            image_path_rel = os.path.relpath(img_path, output_dir)

        # Build sample record
        rec = copy.deepcopy(base_rec)
        md = rec.setdefault("metadata", {})
        md["episode_id"] = episode_id
        md["step_id"] = step_id
        md["pkl_path"] = str(pkl_path)
        # Attach episode-level metadata from task map (src/dest/distractors, etc.)
        if isinstance(episode_meta, dict):
            for k in ("src", "src_distractors", "dest", "dest_distractors"):
                if k in episode_meta:
                    md[k] = episode_meta[k]
        rec["obs_image"] = image_path_rel  # Use relative path for portability
        # Remove obs_text if present (not needed in final training data)
        if "obs_text" in rec:
            del rec["obs_text"]

        steps.append(
            Step(
                episode_id=episode_id,
                step_id=step_id,
                sample=rec,
                action_dict=action_dict,
                obs_text=obs_text_str,
                image_path=image_path_abs,  # Use absolute path for GPT annotation
            )
        )
        step_id += 1

    return steps


def annotate_episode(
    steps: List[Step],
    instruction: str,
    system_prompt: str,
    client: OpenAI,
    episode_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Annotate all steps in an episode."""
    thinking_history: List[str] = []
    # Script-maintained accumulated history items (deterministic, no GPT hallucination)
    accumulated_history_items: List[str] = []
    
    all_samples = []

    # Step 0: virtual instruction-only step
    step0_sample = copy.deepcopy(steps[0].sample if steps else {})
    step0_sample["metadata"]["step_id"] = 0
    step0_sample["obs_image"] = ""
    # Remove obs_text (not needed in training data)
    if "obs_text" in step0_sample:
        del step0_sample["obs_text"]
    
    # Add system prompt (only once)
    convs = step0_sample.setdefault("conversations", [])
    if not any(c.get("from") == "system" for c in convs):
        convs.insert(0, {"from": "system", "value": system_prompt})
    
    object_info = {}
    if isinstance(episode_meta, dict):
        object_info = {
            "src": episode_meta.get("src", ""),
            "src_distractors": episode_meta.get("src_distractors", []),
            "dest": episode_meta.get("dest", ""),
            "dest_distractors": episode_meta.get("dest_distractors", []),
        }

    human_prompt = build_human_prompt_step0(
        instruction=instruction,
        init_action=format_action_dict(steps[0].action_dict),
        object_info=object_info,
    )
    set_human_value(step0_sample, human_prompt)
    
    print(f"[anno] episode={steps[0].episode_id if steps else 'unknown'} step_id=0 request_start")
    thinking_raw = call_chat_thinking(
        client=client,
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        image_path=None,
        temperature=0.0,
    )
    print(f"[anno] episode={steps[0].episode_id if steps else 'unknown'} step_id=0 request_done chars={len(thinking_raw)}")
    
    if not thinking_raw or not thinking_raw.strip():
        print(f"[anno][WARNING] episode={steps[0].episode_id if steps else 'unknown'} step_id=0 GPT returned empty response!")
        thinking_clean = json.dumps({"CoT": "", "Summary": {"History": "", "New Schedule": None, "Current subtask": ""}}, ensure_ascii=False)
    else:
        thinking_clean = strip_special_tokens(thinking_raw)
        json_only = extract_first_json_object(thinking_clean)
        if json_only:
            parsed = parse_json_or_none(json_only)
            if isinstance(parsed, dict):
                thinking_clean = json.dumps(canonicalize_thinking_dict(parsed), ensure_ascii=False)
            else:
                print(f"[anno][WARNING] step_id=0 JSON parse failed, using raw: {json_only[:100]}")
                thinking_clean = json_only
        else:
            print(f"[anno][WARNING] step_id=0 No JSON found in response, first 200 chars: {thinking_clean[:200]}")
            thinking_clean = thinking_raw
    
    set_gpt_thinking(step0_sample, thinking_clean)
    
    # Step 0 action = first PKL action
    if steps:
        action_text = format_action_dict(steps[0].action_dict)
        if action_text:
            set_gpt_action(step0_sample, action_text)
    
    thinking_history.append(thinking_clean)
    # Initialize accumulated history with Step 0 fixed content
    accumulated_history_items.append("1) I have completed a detailed task analysis.")
    all_samples.append(step0_sample)

    # Process remaining steps
    action_history: List[str] = []
    if steps:
        # Initialize with step 0's action
        action_history.append(format_action_dict(steps[0].action_dict))
    
    for idx, st in enumerate(steps):
        obs_dict = {"text": st.obs_text}
        
        # Add system prompt (only once)
        convs = st.sample.setdefault("conversations", [])
        if not any(c.get("from") == "system" for c in convs):
            convs.insert(0, {"from": "system", "value": system_prompt})
        
        # prev_action: last executed action aligned with current observation
        prev_action = format_action_dict(steps[idx].action_dict) if idx < len(steps) else ""
        next_acton = format_action_dict(steps[idx + 1].action_dict) if idx + 1 < len(steps) else format_action_dict({"finish": {}})
        
        # Build accumulated history string from all previous steps
        accumulated_history_str = "\n".join(accumulated_history_items)
        
        human_prompt = build_human_prompt_stepN(
            instruction=instruction,
            prev_action_text=prev_action,
            next_acton=next_acton,
            accumulated_history=accumulated_history_str,
            obs_dict=obs_dict,
            object_info=object_info,
        )
        
        set_human_value(st.sample, human_prompt)
        
        if DEBUG_IMAGE and st.image_path:
            exists = Path(st.image_path).exists()
            print(f"[anno][image] episode={st.episode_id} step_id={st.step_id} using_image={st.image_path} exists={exists}")
        
        print(f"[anno][llm] episode={st.episode_id} step_id={st.step_id} request_start")
        thinking_raw = call_chat_thinking(
            client=client,
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            image_path=st.image_path,
            temperature=0.0,
        )
        print(f"[anno][llm] episode={st.episode_id} step_id={st.step_id} request_done chars={len(thinking_raw)}")
        
        # Debug: print first 300 chars of GPT response
        if thinking_raw:
            print(f"[DEBUG] episode={st.episode_id} step_id={st.step_id} GPT response preview: {thinking_raw[:300]}")
        
        # parsed_thinking will hold the successfully parsed dict for later use
        parsed_thinking: Optional[Dict[str, Any]] = None
        
        if not thinking_raw or not thinking_raw.strip():
            print(f"[anno][WARNING] episode={st.episode_id} step_id={st.step_id} GPT returned empty response!")
            thinking_clean = json.dumps({"CoT": "", "Summary": {"History": "", "New Schedule": None, "Current subtask": ""}}, ensure_ascii=False)
        else:
            thinking_clean = strip_special_tokens(thinking_raw)
            json_only = extract_first_json_object(thinking_clean)
            if json_only:
                parsed = parse_json_or_none(json_only)
                if isinstance(parsed, dict):
                    # Debug: check if parsed JSON has empty fields
                    if not parsed.get("CoT") and not parsed.get("Summary", {}).get("History"):
                        print(f"[DEBUG] episode={st.episode_id} step_id={st.step_id} GPT returned JSON with empty fields!")
                    parsed_thinking = parsed  # Save for later History extraction
                    thinking_clean = json.dumps(canonicalize_thinking_dict(parsed, include_last_action=True), ensure_ascii=False)
                else:
                    print(f"[anno][WARNING] episode={st.episode_id} step_id={st.step_id} JSON parse failed, using raw: {json_only[:100]}")
                    thinking_clean = json_only
            else:
                print(f"[anno][WARNING] episode={st.episode_id} step_id={st.step_id} No JSON found in response, first 200 chars: {thinking_clean[:200]}")
                # Fallback: use raw response
                thinking_clean = thinking_raw
        
        # Extract GPT's Summary.History and append to accumulated_history_items
        # Use parsed_thinking (from successful parse) instead of re-parsing thinking_clean
        if parsed_thinking and isinstance(parsed_thinking, dict) and "Summary" in parsed_thinking:
            s = parsed_thinking["Summary"]
            if isinstance(s, dict):
                gpt_history = s.get("History", "")
                if gpt_history and isinstance(gpt_history, str) and gpt_history.strip():
                    # Append with sequence number
                    step_num = len(accumulated_history_items) + 1
                    accumulated_history_items.append(f"{step_num}) {gpt_history.strip()}")
                else:
                    print(f"[DEBUG] episode={st.episode_id} step_id={st.step_id} GPT returned empty History, skipping append")
                
                # Overwrite GPT's Summary.History with accumulated history for training data
                s["History"] = "\n".join(accumulated_history_items)
                # Re-serialize thinking_clean with updated History
                thinking_clean = json.dumps(canonicalize_thinking_dict(parsed_thinking, include_last_action=True), ensure_ascii=False)
        else:
            print(f"[DEBUG] episode={st.episode_id} step_id={st.step_id} Failed to parse thinking, cannot extract History")
        
        set_gpt_thinking(st.sample, thinking_clean)
        
        # Action = next step's action (or none if last step)
        current_action_text = ""
        if idx + 1 < len(steps):
            current_action_text = format_action_dict(steps[idx + 1].action_dict)
            if current_action_text:
                set_gpt_action(st.sample, current_action_text)
                action_history.append(current_action_text)
        elif idx + 1 == len(steps):
            # Last step: use "finish" action
            finish_action_text = format_action_dict({"finish": {}})
            set_gpt_action(st.sample, finish_action_text)
            action_history.append(finish_action_text)
            
        thinking_history.append(thinking_clean)
        
        all_samples.append(st.sample)

    return all_samples


def load_jobs_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """Load JOBS configuration from a JSON file.

    Relative paths in the config (e.g. template) are resolved relative to the
    config file's directory.
    """
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    with open(config_file, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    # Resolve relative paths
    for scene_id, cfg in jobs.items():
        for key in ("output_dir", "input_data_path", "template"):
            if key in cfg:
                p = Path(cfg[key])
                if not p.is_absolute():
                    cfg[key] = str((config_dir / p).resolve())
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Annotate embodied manipulation trajectories with Chain-of-Thought reasoning."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the jobs configuration JSON file (see config_example.json)."
    )
    parser.add_argument(
        "--limit", type=str, default="all",
        help="Max number of episodes to annotate per job. 'all' for no limit, or an integer (default: all)."
    )
    args = parser.parse_args()

    JOBS = load_jobs_config(args.config)
    annotation_limit = args.limit

    def _episode_id_sort_key(ep: str) -> int:
        m = re.match(r"^ep_(\d+)$", str(ep).strip())
        return int(m.group(1)) if m else 10**18

    def run_episode(
        scene_id: str,
        pkl_path: Path,
        episode_id: str,
        instruction: str,
        base_rec: Dict[str, Any],
        system_prompt: str,
        output_dir: Path,
        checkpoint_dir: Path,
        episode_meta: Dict[str, Any],
    ) -> Tuple[int, str, List[Dict[str, Any]]]:
        ep_idx = _episode_id_sort_key(episode_id)
        print(f"[{scene_id}] Processing episode {episode_id} ({pkl_path.name})")

        steps = process_pkl_to_steps(
            pkl_path, episode_id, output_dir, base_rec, episode_meta=episode_meta
        )
        print(f"[{scene_id}] episode {episode_id}: {len(steps)} steps extracted")

        client = make_openai_client()
        annotated = annotate_episode(
            steps, instruction, system_prompt, client, episode_meta=episode_meta
        )

        ckpt_path = checkpoint_dir / f"annotated_checkpoint.{episode_id}.json"
        save_json(str(ckpt_path), annotated)
        print(f"[{scene_id}] checkpoint saved: {ckpt_path}")

        return ep_idx, episode_id, annotated

    for scene_id, cfg in JOBS.items():
        input_data_path = Path(cfg["input_data_path"])
        output_dir = Path(cfg["output_dir"])
        template_path = Path(cfg["template"])

        if not input_data_path.exists():
            print(f"[{scene_id}] skip: input data path missing -> {input_data_path}")
            continue
        if not template_path.exists():
            print(f"[{scene_id}] skip: template missing -> {template_path}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "anno_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Scan PKL files in input_data_path and load corresponding meta files
        episodes: List[Tuple[str, Path, Dict[str, Any]]] = []

        # Filter out error PKL files and sort
        all_pkl_files = sorted(input_data_path.glob("*.pkl"))
        pkl_files = [p for p in all_pkl_files if not p.stem.startswith("error")]

        if len(all_pkl_files) != len(pkl_files):
            filtered_count = len(all_pkl_files) - len(pkl_files)
            print(f"[{scene_id}] filtered out {filtered_count} error PKL files")

        print(f"[{scene_id}] processing {len(pkl_files)} valid PKL files")

        # Process each PKL file and assign sequential episode IDs starting from 1
        for idx, pkl_path in enumerate(pkl_files, start=1):
            # Assign sequential episode_id (e.g., "ep_000001", "ep_000002", ...)
            episode_id = f"ep_{idx:06d}"
            pkl_stem = pkl_path.stem  # Keep original stem for meta file lookup

            # Load corresponding meta file (e.g., "203_meta.json")
            meta_path = pkl_path.parent / f"{pkl_stem}_meta.json"

            if not meta_path.exists():
                print(f"[{scene_id}] warning: meta file missing for {pkl_path.name} -> {meta_path}")
                continue

            try:
                meta = load_json(str(meta_path))

                # Extract task_description as instruction
                instruction = meta.get("task_description", "").strip()
                if not instruction:
                    print(f"[{scene_id}] skip {episode_id}: missing task_description in meta")
                    continue

                # Build episode metadata
                # src_distractors and dest_distractors are already in meta.json as furniture lists
                episode_meta = {
                    "task_instruction": instruction,
                    "src": meta.get("src", ""),
                    "dest": meta.get("dest", ""),
                    "src_distractors": meta.get("src_distractors", []),
                    "dest_distractors": meta.get("dest_distractors", []),
                }

                episodes.append((episode_id, pkl_path, episode_meta))

            except Exception as e:
                print(f"[{scene_id}] error loading meta for {pkl_path.name}: {e}")
                continue

        print(f"[{scene_id}] found {len(episodes)} episodes with valid meta files")

        # Sort episodes by episode_id numeric
        episodes = sorted(episodes, key=lambda t: _episode_id_sort_key(t[0]))

        # Apply annotation limit if set
        if annotation_limit != "all":
            limit = int(annotation_limit)
            episodes = episodes[:limit]
            print(f"[{scene_id}] limiting to first {limit} episodes for testing")

        # Load template
        base_rec = load_template_record(template_path)
        system_prompt = find_system_prompt(base_rec)

        print(f"[{scene_id}] processing {len(episodes)} episodes")
        
        futures = []
        with ThreadPoolExecutor(max_workers=EPISODE_WORKERS) as executor:
            for episode_id, pkl_path, meta in episodes:
                instruction = str(meta.get("task_instruction", "") or "").strip()
                if not instruction:
                    print(f"[{scene_id}] skip episode {episode_id}: missing task_instruction")
                    continue
                if not pkl_path.exists():
                    print(f"[{scene_id}] skip episode {episode_id}: pkl missing -> {pkl_path}")
                    continue
                futures.append(
                    executor.submit(
                        run_episode,
                        scene_id,
                        pkl_path,
                        episode_id,
                        instruction,
                        base_rec,
                        system_prompt,
                        output_dir,
                        checkpoint_dir,
                        meta,
                    )
                )

            results: List[Tuple[int, str, List[Dict[str, Any]]]] = []
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"[{scene_id}] episode worker failed: {e}")

        if not results:
            print(f"[{scene_id}] no episodes processed")
            continue

        results_sorted = sorted(results, key=lambda x: x[0])
        all_annotated: List[Dict[str, Any]] = []
        for _, episode_id, annotated in results_sorted:
            all_annotated.extend(annotated)
        
        # Save final output
        output_json = output_dir / "annotated_output.json"
        save_json(str(output_json), all_annotated)
        print(f"[{scene_id}] saved: {output_json}, total samples: {len(all_annotated)}")


if __name__ == "__main__":
    main()

