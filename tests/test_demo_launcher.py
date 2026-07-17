"""Regression tests for the public demo launcher environment contract."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "demo" / "run_mcp_server_demo.sh"


def test_launcher_loads_env_file_without_overwriting_caller(tmp_path):
    env_file = tmp_path / "demo.env"
    env_file.write_text(
        "OPENAI_API_KEY=from-file\n"
        "OPENAI_API_BASE_URL=https://file.example/v1\n"
        "OPENAI_MODEL=file-model\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "captured.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$OPENAI_API_KEY" "$OPENAI_API_BASE_URL" '
        '"$OPENAI_MODEL" "$AUTO_LOAD_EPISODE" "$PWD" "$DEMO_TASK_CONFIG" '
        '> "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ENV_FILE": str(env_file),
            "CAPTURE_PATH": str(capture),
            "OPENAI_API_KEY": "from-caller",
            "AUTO_LOAD_EPISODE": "1",
            # Isaac Sim's conda hook exports this misleading value.  The
            # launcher must still locate the REAL checkout through $0.
            "BASH_SOURCE": "/isaac-sim/setup_python_env.sh",
        }
    )
    env.pop("OPENAI_API_BASE_URL", None)
    env.pop("OPENAI_MODEL", None)

    subprocess.run(
        [str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "from-caller",
        "https://file.example/v1",
        "file-model",
        "1",
        str(REPO_ROOT),
        str(REPO_ROOT / "configs" / "demo_task.yaml"),
    ]
