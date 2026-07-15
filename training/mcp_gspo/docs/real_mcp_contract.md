# REAL MCP Contract

## Baseline

- Repository: `git@github.com:InternRobotics/REAL.git`
- Commit: `e26838d723e3831a53dcabf2e16ba494a5aa1c25`
- Source of truth: `mcp_server/tools.py`, `mcp_server/actions.py`, and
  `mcp_server/mcp_server_demo.py`

The runtime checks the ten public action names and their required argument keys
after every MCP reset. Set `MCP_STRICT_REAL_CONTRACT=0` only while migrating an
older private server.

## Public Actions

| Tool | Required arguments |
| --- | --- |
| `list_receptacles` | none |
| `navigate_to` | `receptacle_name` |
| `explore_receptacle` | none |
| `focus_on` | `marker_id` |
| `find_objects` | `target_category` |
| `highlight_receptacles` | none |
| `pick` | `marker_id` |
| `place` | `marker_id` |
| `open` | `marker_id` |
| `close` | `marker_id` |

Action responses may contain text and images. The final text item should use the
REAL debug form `Debug Info:\n<json>`. The client consumes `CURRENT_INV`,
`CURRENT_MARKER_MAP`, `CURRENT_LANDMARK`, and the lowercase `world_graph` field.

## Reset Payload

The training environment calls `finish({})`, or
`finish_with_id({"task_id": ...})` under explicit task scheduling. Its first text
response must be a JSON object containing:

- `task_id`
- `task_description`
- `initial_world_graph`
- `target_world_graph` or `goal_world_graph`
- `rooms_and_furniture`
- `target_object` with `id` and `category`
- `source_furniture`
- `destination_furniture`

`list_tasks({})` must return a JSON list of task IDs for each scheduled worker.
The public REAL demo server supports sequential `finish` task loading but does
not advertise the training-only lifecycle tools through `list_tools`.

## Compatibility Boundary

Legacy action aliases are accepted only in model output and are resolved to a
discovered server tool. Rewards, metrics, logs, and prompt text use canonical
REAL names. This keeps old checkpoints usable without carrying the old server
contract into new code.
