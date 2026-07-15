# Contributing to REAL

Thank you for helping improve REAL.

## Development workflow

1. Fork the repository and create a focused branch.
2. Keep machine-specific paths, credentials, datasets, assets, and model weights out of commits.
3. Install and enable the formatting hooks once per clone:

   ```bash
   python -m pip install pre-commit
   pre-commit install
   ```

4. Run the formatting and file-integrity checks before opening a pull request:

   ```bash
   pre-commit run --all-files
   ```

5. Run the lightweight behavior checks:

   ```bash
   python -m pytest -q tests
   python -m compileall -q mcp_server proc_datagen
   bash -n scripts/demo/run_mcp_server_demo.sh
   bash -n scripts/filter/batch_filter_proc.sh
   bash -n training/qwen3vl_sft/train_qwen3vl_sft.sh
   ```

6. Describe the affected workflow and any external assets needed to reproduce the change.

## Reporting issues

Include the operating system, Python version, Isaac Sim and InternUtopia versions, the command you ran, and the complete error traceback. Do not include API keys or private data paths.
