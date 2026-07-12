# Qwen3-VL SFT Training

This folder contains the public REAL Qwen3-VL SFT training entrypoint.

REAL does not vendor the full Qwen3-VL training framework. Use the official Qwen3-VL fine-tuning code as the trainer implementation, and use this folder for REAL data examples, launch templates, and configuration notes.

## Files

- `train_qwen3vl_sft.sh`: public launch template for Qwen3-VL SFT.
- `datasets.example.yaml`: example REAL dataset names and path layout.
- `deepspeed/zero2.json`: ZeRO-2 DeepSpeed config following the official Qwen3-VL finetune template shape.
- `minimal_dataset.json`: minimal image + conversations training example.

## Upstream Qwen3-VL Code and Environment

Before reproducing REAL SFT, first follow the official Qwen3-VL fine-tuning guide to configure the training environment. This folder does not include the Qwen3-VL trainer, model implementation, CUDA setup, FlashAttention setup, or dependency lockfile.

Clone the official Qwen3-VL repository and use its `qwen-vl-finetune` directory as the training root:

```bash
git clone https://github.com/QwenLM/Qwen3-VL.git
export QWEN3VL_FINETUNE_ROOT=/path/to/Qwen3-VL/qwen-vl-finetune
```

Install the environment following the official Qwen3-VL fine-tuning documentation. The exact CUDA, PyTorch, FlashAttention, DeepSpeed, Transformers, and qwen-vl-utils versions should follow the upstream instructions for your hardware.

Use this REAL folder only after the official Qwen3-VL fine-tuning example can run in your environment.

## Base Model

The REAL SFT template is based on the Hugging Face model [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Download the base model before launching training:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/Qwen3-VL-8B-Instruct
```

Then point the training script to the downloaded model directory:

```bash
export MODEL_NAME_OR_PATH=models/Qwen3-VL-8B-Instruct
```

You can also keep `MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-8B-Instruct` if your environment is allowed to download from Hugging Face at runtime.

## Data Format

Training examples should follow the Qwen-VL chat-style JSON format. Each item contains media paths plus `conversations` turns:

```json
{
  "image": "images/example_0001.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat action should the agent take next?"
    },
    {
      "from": "gpt",
      "value": "The agent should navigate to the table and pick up the marked cup."
    }
  ]
}
```

See `minimal_dataset.json` for a minimal example and `datasets.example.yaml` for suggested public dataset names.

## Register REAL Datasets

The official Qwen3-VL trainer resolves dataset names through `qwenvl/data/__init__.py` in the upstream fine-tuning repository. Add entries that point to your prepared REAL annotations and media root.

Example:

```python
REAL_BASIC_PNP = {
    "annotation_path": "/path/to/real_sft/basic_pnp/annotations.json",
    "data_path": "/path/to/real_sft/basic_pnp",
}

REAL_ARTICULATION = {
    "annotation_path": "/path/to/real_sft/articulation/annotations.json",
    "data_path": "/path/to/real_sft/articulation",
}

data_dict.update(
    {
        "real_basic_pnp": REAL_BASIC_PNP,
        "real_articulation": REAL_ARTICULATION,
    }
)
```

You can also append a sampling percentage when launching training, such as `real_basic_pnp%50`.

## Launch Training

From the REAL repository root:

```bash
export QWEN3VL_FINETUNE_ROOT=/path/to/Qwen3-VL/qwen-vl-finetune
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/Qwen3-VL-8B-Instruct
export MODEL_NAME_OR_PATH=models/Qwen3-VL-8B-Instruct
export DATASETS=real_basic_pnp,real_articulation
export OUTPUT_DIR=outputs/qwen3vl_sft

bash training/qwen3vl_sft/train_qwen3vl_sft.sh
```

The launch script uses the following public configuration variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3VL_FINETUNE_ROOT` | required | Path to the official `qwen-vl-finetune` directory |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face model id or local downloaded model path |
| `DATASETS` | `real_basic_pnp` | Comma-separated dataset names registered upstream |
| `OUTPUT_DIR` | `outputs/qwen3vl_sft` | Training output directory |
| `DEEPSPEED_CONFIG` | `training/qwen3vl_sft/deepspeed/zero2.json` | DeepSpeed config path |
| `NPROC_PER_NODE` | `1` | Number of local GPU processes |
| `MASTER_ADDR` | `127.0.0.1` | Distributed launch address |
| `MASTER_PORT` | `29500` | Distributed launch port |

The defaults in `train_qwen3vl_sft.sh` are a sanitized version of the internal 8B v7 SFT recipe:

- base model: `Qwen/Qwen3-VL-8B-Instruct`
- training length: 4 epochs
- per-device batch size: 2
- gradient accumulation: 4
- learning rate: `5e-6`
- warmup ratio: `0.08`
- max pixels: `100352`
- model max length: `8192`
- save strategy: `epoch`
- save total limit: 5
- default logging backend: `none`

Common training hyperparameters can also be overridden:

```bash
export NPROC_PER_NODE=8
export LEARNING_RATE=5e-6
export NUM_TRAIN_EPOCHS=4
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4
export WARMUP_RATIO=0.08
export MODEL_MAX_LENGTH=8192
export MAX_PIXELS=100352
```

## Notes for Public Release

- This repository publishes launch templates and data configuration examples only.
- Reproduction requires cloning the official Qwen3-VL repository and setting up its official fine-tuning environment first.
- It does not include private cluster launch files, private data, trained weights, local cache layouts, or service credentials.
- The included ZeRO-2 config follows the official Qwen3-VL fine-tuning template shape. Check the upstream repository for the latest recommended DeepSpeed settings.
