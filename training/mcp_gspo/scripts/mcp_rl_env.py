"""Compatibility entrypoint for ms-swift's ``--external_plugins`` option.

The implementation lives in :mod:`rl_runtime.swift_plugin`; keeping this thin
file preserves existing launch commands while making the runtime importable and
testable as a normal package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_runtime.swift_plugin import *  # noqa: F401,F403,E402
