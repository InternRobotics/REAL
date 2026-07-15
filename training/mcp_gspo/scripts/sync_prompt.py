#!/usr/bin/env python3
"""Validate and synchronize a local or SSH-hosted rollout prompt."""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_runtime.prompt import DEFAULT_PROMPT_PATH, validate_prompt_template  # noqa: E402

SSH_SOURCE = re.compile(r"^(?P<host>[A-Za-z0-9_.@-]+):(?P<path>/.*)$")
ASSIGNMENT = re.compile(
    r"PROMPT_TEMPLATE\s*=\s*(?P<quote>\"\"\"|''')(?P<body>.*?)(?P=quote)",
    re.DOTALL,
)


def read_source(source: str) -> str:
    local_path = Path(source).expanduser()
    if local_path.is_file():
        return local_path.read_text(encoding="utf-8")

    match = SSH_SOURCE.fullmatch(source)
    if not match:
        raise ValueError(f"prompt source is neither a file nor host:/absolute/path: {source}")
    command = f"cat -- {shlex.quote(match.group('path'))}"
    result = subprocess.run(
        ["ssh", match.group("host"), command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_template(content: str) -> str:
    match = ASSIGNMENT.search(content)
    if match:
        content = match.group("body")
    return content.strip() + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local prompt file or SSH source host:/absolute/path")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help=f"managed prompt path (default: {DEFAULT_PROMPT_PATH})",
    )
    parser.add_argument("--apply", action="store_true", help="replace the managed prompt")
    args = parser.parse_args()

    candidate = extract_template(read_source(args.source))
    validate_prompt_template(candidate)

    destination = args.destination.resolve()
    current = destination.read_text(encoding="utf-8") if destination.exists() else ""
    if current.rstrip() == candidate.rstrip():
        print(f"prompt is synchronized: {destination}")
        return 0

    diff = difflib.unified_diff(
        current.splitlines(),
        candidate.splitlines(),
        fromfile=str(destination),
        tofile=args.source,
        lineterm="",
    )
    print("\n".join(diff))
    if not args.apply:
        print("prompt differs; rerun with --apply after reviewing the diff")
        return 1

    atomic_write(destination, candidate)
    print(f"updated prompt: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
