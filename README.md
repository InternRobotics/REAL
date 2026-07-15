<p align="center">
  <img src="docs/real-mark-v2.png" alt="REAL logo" width="160" />
</p>

<h1 align="center">REAL: Exploratory, Communicative, and Deployable Embodied Agents</h1>

<p align="center">
  <a href="#dataset--checkpoint"><img src="https://img.shields.io/badge/Dataset%20%26%20Checkpoint-Coming%20Soon-f59e0b?style=flat-square" alt="Dataset and Checkpoint" /></a>
  <a href="#paper"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-3b82f6?style=flat-square" alt="Paper" /></a>
  <a href="https://internrobotics.github.io/REAL/"><img src="https://img.shields.io/badge/Project%20Page-Website-10b981?style=flat-square" alt="Project Page" /></a>
  <a href="https://github.com/InternRobotics/REAL"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="Code" /></a>
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" alt="MIT License" />
</p>

---

## 📰 News

* **[2026.06.18]** 🎉 Our paper has been accepted to **ECCV 2026**! 🥳
* **[2026.06.01]** 🚀 Training code released.
* **[2026.03.31]** Procedural task generation and trajectory annotation utilities released.
* **[2026.03.24]** Simulation environment and MCP server released.

---

## Introduction

**REAL** is a sim-to-real-consistent framework for interactive open-world mobile manipulation. Agents explore from raw RGB observations, use deployable navigation and manipulation tools, and communicate with a simulated user to resolve ambiguous instructions without privileged simulator information.

### Contributions

* **REAL framework**: Non-privileged visual exploration with interactive intent alignment and an MCP-based tool interface.
* **Training and benchmark**: A hierarchical SFT and online RL pipeline evaluated on REAL-Bench, which contains 241 tasks across four task families.
* **Sim-to-real deployment**: 56.9% success on interactive tasks and 78.3% success over 60 real-world robot episodes.

### Repository layout

| Path | Purpose |
|------|---------|
| `mcp_server/` | MCP tools, server, perception utilities, and simulation environment setup |
| `configs/` | Portable demo task configuration |
| `proc_datagen/` | Procedural task generation, annotation, and physics verification |
| `training/qwen3vl_sft/` | Public Qwen3-VL SFT launch and dataset templates |
| `scripts/` | Demo and batch-processing entrypoints |

### Online RL branch

The MCP-based online GRPO runtime is maintained on the [`gspo`](../../tree/gspo)
branch under `training/mcp_gspo/`. It is kept separate from `main` because it
depends on the ms-swift rollout stack and external MCP workers rather than the
public demo runtime. To use it, fetch the branch and switch explicitly:

```bash
git fetch origin gspo
git switch gspo
```

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `list_receptacles` | List all receptacles by room |
| `navigate_to` | Navigate to a furniture receptacle |
| `explore_receptacle` | Survey all objects on the current receptacle |
| `focus_on` | Focus camera on a specific object by marker ID |
| `find_objects` | Find and highlight objects of a given category in view |
| `highlight_receptacles` | Highlight all visible receptacle surfaces |
| `pick` | Pick up an object by marker ID |
| `place` | Place held object onto a receptacle surface |
| `open` / `close` | Operate articulated doors |


Each tool call returns an RGB observation image and structured text feedback from the simulation.

---

## Quick Start

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/InternRobotics/REAL.git
cd REAL
```

### 2. Install InternUtopia

Please refer to the InternUtopia [documentation](https://internrobotics.github.io/user_guide/internutopia/get_started/installation.html).

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

Optional Qwen3-VL training dependencies are managed by the upstream Qwen3-VL fine-tuning environment rather than this runtime requirements file.

### 4. Download and unpack assets

Download `assets.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/15RXHNisGn5SZTLvFWYkdKKazNvxaRrVd?usp=sharing) and extract it into the repository root.

After extraction, the `assets/` directory should contain the scenes, models, objects, materials, and metadata required by the demo.

Copy `.env.example` to `.env` only when you need an OpenAI-compatible perception endpoint. Never commit the populated `.env` file.

### 5. Run the demo MCP server

```bash
./scripts/demo/run_mcp_server_demo.sh
```

The server binds to `127.0.0.1:8080` by default. Override it with `HOST=<host>` and `PORT=<port>`, then connect an MCP-compatible agent to `http://127.0.0.1:8080/sse`.


## Task Generation Pipeline

The procedural task generation pipeline lives in `proc_datagen/`. It produces pick-and-place task configs for training and evaluation, in two stages:

```mermaid
flowchart LR
    A["task_generator.py\n(task gen + static filter)"]
    -->|"YAML per\nscene/type"| B["verify_proc.py\n(physics simulation)"]
    B -->|pass| C["physics_passed.yaml"]
    B -->|fail| D["physics_failed.yaml"]
    A -->|"--to-json"| E["JSON files\n(backward compat)"]
    A -->|"--polish"| A
```

### Task types

| Type | Description |
|------|-------------|
| `basic` | Simple pick-and-place with same-type furniture distractors |
| `distractor` | Same-category object distractors; uses `detailed_caption` for grounding |
| `articulation` | Store / retrieve involving articulated furniture (open/close door) |
| `interactive` | Same-purpose different-category distractors + fuzzy description (requires user interaction to disambiguate) |
| `gather` | Multi-source gather: collect N objects to one destination |

### Asset setup — MesaTask USD files

The task generator relies on object USD files from the [MesaTask dataset](https://huggingface.co/datasets/InternRobotics/MesaTask-10K).  After downloading, set `MESATASK_USD_ROOT` to the directory containing the `.usd` files before running any pipeline script:

```bash
export MESATASK_USD_ROOT=/path/to/mesatask_download/object_usds
```

The metadata file `assets/metadata/consolidated_asset_library_with_size.json` stores only filenames (e.g. `abc123.usd`); the code resolves them against `MESATASK_USD_ROOT` at runtime.

### Stage 1 — Task generation & static filtering

```bash
# Generate all 5 task types, with inline static placement check
# Output: proc_datagen/configs/{scene_id}/{task_type}.yaml
python proc_datagen/task_generator.py \
    --tasks all \
    --output-dir proc_datagen/configs \
    --verify-placement \
    --occ-map-root assets/metadata \
    --seed 42

# Generate only specific types
python proc_datagen/task_generator.py \
    --tasks interactive gather \
    --output-dir proc_datagen/configs

# Polish task descriptions with an LLM after generation
# (requires OPENAI_API_KEY and openai package)
python proc_datagen/task_generator.py \
    --tasks all \
    --output-dir proc_datagen/configs \
    --verify-placement \
    --polish

# Also export flat JSON files (backward compat)
python proc_datagen/task_generator.py \
    --tasks all \
    --output-dir proc_datagen/configs \
    --verify-placement \
    --to-json
```

Output: `proc_datagen/configs/{scene_id}/{task_type}.yaml` — per-scene per-type YAML files containing `objects` (with positions) and `episodes` (with placements).

### Stage 2 — Physics verification

Run physics simulation to filter out tasks where objects fall or leave the surface:

The provided batch script processes `articulation`, `interactive`, `distractor`, and `gather`. The `basic` task type can be checked manually with `verify_proc.py` using the same environment variables shown below.

```bash
# Verify all scenes and merge results (default)
./scripts/filter/batch_filter_proc.sh

# Only run physics (skip merge)
./scripts/filter/batch_filter_proc.sh --stage physics

# Only merge already-finished results
./scripts/filter/batch_filter_proc.sh --stage merge
```

Results per task type:

```
proc_datagen/verify_results/{task_type}/
    physics_valid.yaml                     # merged passing episodes across all scenes
    {scene_id}/physics_passed.yaml         # per-scene passing episodes
    {scene_id}/physics_failed.yaml         # per-scene failed episodes
```

To run a single scene manually (e.g. for debugging):

```bash
TASK_SOURCE_PATH=proc_datagen/configs/MVUCSQAKTKJ5EAABAAAAABQ8/interactive.yaml \
OUTPUT_PATH=proc_datagen/verify_results/interactive/MVUCSQAKTKJ5EAABAAAAABQ8 \
python proc_datagen/verify_proc.py --max-tasks 20
```

---

## Trajectory Annotation

`proc_datagen/trajectory_annotation/` converts existing replay PKL files and their metadata into step-level JSON annotations using an OpenAI-compatible multimodal endpoint. It annotates previously recorded trajectories; it does not record trajectories itself.

Provide `OPENAI_API_KEY` and, when needed, `OPENAI_BASE_URL` and `OPENAI_MODEL`. Create a job configuration based on `proc_datagen/trajectory_annotation/config_example.json`, then run:

```bash
python proc_datagen/trajectory_annotation/annotate_trajectory.py \
    --config /path/to/trajectory_annotation_config.json
```

Do not commit credentials, private replay data, or machine-specific paths.

---

## Qwen3-VL SFT Training

REAL provides public launch templates and dataset configuration examples for supervised fine-tuning on top of the official Qwen3-VL fine-tuning workflow. Reproduction requires cloning the official Qwen3-VL repository and setting up its official fine-tuning environment first.

See [training/qwen3vl_sft/README.md](training/qwen3vl_sft/README.md) for the full training guide, launch script, data config example, DeepSpeed config, and minimal dataset example.

The template entrypoint is:

```bash
git clone https://github.com/QwenLM/Qwen3-VL.git
export QWEN3VL_FINETUNE_ROOT=/path/to/Qwen3-VL/qwen-vl-finetune
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/Qwen3-VL-8B-Instruct
export MODEL_NAME_OR_PATH=models/Qwen3-VL-8B-Instruct
export DATASETS=real_basic_pnp

bash training/qwen3vl_sft/train_qwen3vl_sft.sh
```

This branch does not publish private cluster scripts, internal data paths,
service credentials, or model weights. The version-controlled online RL runtime
is available on the `gspo` branch; private deployment topology and credentials
remain excluded there as well.

---


## 📑 Citation

The paper citation will be added after publication. Until then, please cite this repository URL and the paper title:

> Exploratory, Communicative, and Deployable: Vision-Driven Embodied Agents for Open-World Mobile Manipulation.

---

## Resources

### Dataset & Checkpoint

The dataset and model checkpoint links will be added here after release.

### Paper

The paper link will be added here after publication.

### Project Page

[REAL project page](https://internrobotics.github.io/REAL/)

---

## Acknowledgement

REAL is built on top of [**InternUtopia**](https://github.com/InternRobotics/InternUtopia).

We thank the teams behind [**Model Context Protocol**](https://modelcontextprotocol.io/) and [**NVIDIA Isaac Sim**](https://developer.nvidia.com/isaac-sim) for their foundational work.

---

## License

This project is licensed under the [MIT License](LICENSE).
