"""Regression tests for the public Isaac Sim installation contract."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
README = REPO_ROOT / "README.md"

TESTED_MCP_STACK = {
    "mcp": "1.9.4",
    "pydantic": "2.9.2",
    "starlette": "0.45.3",
    "uvicorn": "0.29.0",
    "sse-starlette": "2.1.3",
    "httpx": "0.28.1",
    "openai": "2.45.0",
}


def _exact_pins() -> dict[str, str]:
    pins = {}
    for raw_line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_requirements_pin_cross_machine_mcp_stack():
    pins = _exact_pins()
    assert {name: pins.get(name) for name in TESTED_MCP_STACK} == TESTED_MCP_STACK


def test_readme_documents_isaac_shell_and_dependency_conflict():
    readme = README.read_text(encoding="utf-8")

    assert "bash -lc" in readme
    assert "mcp==1.9.4" in readme
    assert "httpx==0.25.2" in readme
    assert "drive.google.com/drive/folders/15RXHNisGn5SZTLvFWYkdKKazNvxaRrVd" not in readme
