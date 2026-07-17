# REAL-Bench

This directory contains the official release of 241 REAL-Bench task definitions:

| Family | Tasks |
|---|---:|
| FDP | 72 |
| FODP | 56 |
| FDO | 48 |
| SUL | 65 |
| **Total** | **241** |

Every episode is stored as a directly loadable eval-server config:

```text
tasks/
  FDP/<task_id>.yaml
  FODP/<task_id>.yaml
  FDO/<task_id>.yaml
  SUL/<task_id>.yaml
```

Each YAML is a root mapping with `scene_id`, `paths`, `objects`, and a
single-entry `episodes` list. The episode retains its full benchmark record;
`benchmark_task_id` is the globally unique identifier.

The bundle contains task definitions and `mesa_required.txt`, a sorted lock of
the 287 referenced MesaTask USD basenames. It does not include scene assets,
MesaTask object USDs, trajectories, training data, or model weights. Runtime
object paths use `${MESATASK_USD_ROOT}` and scene paths use the portable
`assets/` layout documented in the repository README.

From the repository root, validate and load all tasks with:

```bash
python -m real_bench
```

Load one episode directly with the same parser used by the eval server:

```python
from mcp_server.config import load_task_config

config = load_task_config("benchmark/tasks/FDP/verify_task_1.yaml")
```

After installing the required assets, the server accepts the same path without
conversion:

```bash
export DEMO_TASK_CONFIG=benchmark/tasks/FDP/verify_task_1.yaml
./scripts/demo/run_mcp_server_demo.sh
```

See the repository [README](../README.md#real-bench-usage) for programmatic
loading and family filtering.
