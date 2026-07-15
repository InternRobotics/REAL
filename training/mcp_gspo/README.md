# REAL MCP GRPO Runtime

This directory contains the online RL runtime for REAL and is maintained on the
`gspo` branch. It runs model rollouts against external REAL MCP servers and
trains with ms-swift GRPO or NGRPO.

## Layout

- `rl_runtime/`: MCP environment, reward implementation, scheduler, and NGRPO operator.
- `scripts/`: rollout, trainer, topology, prompt-sync, and preflight entrypoints.
- `config/`: portable runtime and SSH topology examples.
- `prompts/`: canonical REAL tool-calling prompt.
- `docs/`: ms-swift reroute and REAL MCP contract notes.

The public REAL action schemas are synchronized with the `main` branch at
commit `e26838d723e3831a53dcabf2e16ba494a5aa1c25`. Training-only lifecycle
tools (`finish_with_id` and `list_tasks`) are intentionally handled outside the
public action contract.

## Setup

```bash
cd training/mcp_gspo
conda env create -f environment.yml
conda activate mcp-grpo-rl
pip install -e .
cp config/rl_training.env.example config/rl_training.env
```

Set `MODEL_PATH` and `MCP_SERVER_URLS` in the local configuration. No private
hosts, credentials, logs, checkpoints, or deployment paths are versioned.

## Reward Path

`MCPSingleEnv` computes each environment reward after the MCP action.
`MCPMultiTurnScheduler` sums those step rewards into
`RolloutOutput.rollout_infos["total_reward"]`. The rollout launcher enables
ms-swift gym mode, and ms-swift consumes that exact field as the GRPO reward.
`MCPGymEnv` is also available for synchronous Gymnasium consumers and returns
the same reward without recomputation.

Check this contract before launching a job:

```bash
python scripts/preflight.py --check-reward-contract --import-plugin
```

## NGRPO

NGRPO is implemented in `rl_runtime/ngrpo.py`. Before an NGRPO run, route the
supported ms-swift version to the repository implementation:

```bash
python scripts/patch_ms_swift_ngrpo.py
python scripts/patch_ms_swift_ngrpo.py --apply
```

The patcher is version-gated, idempotent, and keeps a backup of each modified
ms-swift file. See `docs/ms_swift_ngrpo.md` for the modified integration points.

## Run

```bash
bash scripts/start_rollout_server.sh
bash scripts/start_grpo_trainer.sh
```

Use `bash scripts/launch_grpo.sh` to start both process groups together. The
rollout server and trainer normally need separate GPUs; the single-GPU config
is only for one-process debugging after memory has been measured.
