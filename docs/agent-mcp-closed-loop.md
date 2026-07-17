# Agent ↔ MCP closed-loop validation

This repository includes a deterministic one-episode MCP server for validating
the complete client protocol without requiring Isaac Sim. It exposes the same
public MCP tool schemas as the demo server and returns an RGB image, user-facing
text, and structured `Debug Info` after every action.

The validation task requires an agent to move `apple_0` from
`source_counter` to `target_table`. A passing run proves all of the following:

1. the client connects through MCP SSE and discovers the live tool schemas;
2. the client requests and parses the first task through `finish`;
3. model decisions become MCP calls with schema-compatible arguments;
4. RGB observations and debug world graphs return to the policy;
5. `pick` and `place` change the server-owned world graph;
6. the final `finish` is scored successfully and writes trajectory artifacts.

The score is a client-side protocol diagnostic computed from the latest
server-provided world graph. It compares both object contents and any goal door
state. The server remains authoritative for action execution and state changes,
and `finish` advances its episode queue.

This validation path uses its own synthetic episode and does not start the
Isaac Sim server. The released benchmark YAML files are directly compatible
with the simulator eval server through `DEMO_TASK_CONFIG`; full execution still
requires the corresponding scene and MesaTask assets.

## Automated network test

The regression test starts a real HTTP/SSE MCP server on a free loopback port
and runs both policy adapters through the seven-action task. Inference responses
are deterministic in this test so it remains fast and credential-free; MCP,
image transport, policy parsing, state transitions, scoring, and artifact
writing are not mocked.

```bash
python -m unittest tests.test_agents_mcp_e2e
```

Both cases must report one finished episode, one success, zero errors, and a
100% success rate.

## Manual run with a real model

Start a fresh validation server for each agent. Use one agent per server
process; the demo task manager is intentionally single-client.

```bash
PORT=8765 \
VALIDATION_LOG_PATH=/tmp/real-agent-mcp-calls.json \
python -m tests.mcp_validation_server
```

Then run either backend in a second terminal.

### OpenAI-compatible VLM API

```bash
export MODEL_NAME=gpt-4o-mini
export MCP_SERVER_URL=http://127.0.0.1:8765/sse
export EVAL_OUTPUT_PATH=/tmp/real-vlm-validation
export OPENAI_API_KEY=your-key
# export OPENAI_API_BASE_URL=https://your-compatible-endpoint.example/v1

python -m agents.vlm_api_agent
```

If the host defines a SOCKS `ALL_PROXY` but its Python environment does not
include `socksio`, either install the HTTPX SOCKS extra or clear only the SOCKS
fallback for this process while retaining any configured HTTP/HTTPS proxy:

```bash
all_proxy= ALL_PROXY= python -m agents.vlm_api_agent
```

### Local Qwen VLM

`MODEL_PATH` is local-only: the loader never downloads missing model files.
Install a CUDA-compatible PyTorch build before the optional Qwen packages.

```bash
pip install -r requirements-qwen.txt

export MODEL_PATH=/path/to/local/qwen-model
export MCP_SERVER_URL=http://127.0.0.1:8765/sse
export EVAL_OUTPUT_PATH=/tmp/real-qwen-validation
export CUDA_VISIBLE_DEVICES=0

python -m agents.qwen_agent
```

With `accelerate`, Qwen uses `device_map="auto"`. Without it, the agent places
the complete model on one CUDA device when available, otherwise on CPU.

## Evidence audit

After a passing run:

- the agent result JSON has `total=1`, `finished=1`, `successes=1`, `errors=0`,
  and `success_rate=1.0`;
- `episode_summary.json` has `status="finished"` and `success=true`;
- each non-finish action directory contains `tool_call.json`, the raw model response,
  `image_observation_0.png`, and `debug_info.json`;
- the validation call log has `completed=true` and ends with:

```json
{
  "source_counter": {"content": []},
  "target_table": {"content": ["apple_0"]}
}
```

This validation server proves the agent/MCP transport and control loop. It does
not replace simulator evaluation of navigation, perception, manipulation, or
physics in the InternUtopia/Isaac Sim environment.
