# ms-swift NGRPO Reroute

## Verified Target

- Distribution: `ms-swift==3.12.4`
- Interpreter: the Python selected by `PYTHON_BIN`
- Local operator: `rl_runtime.ngrpo.compute_ngrpo_advantages`

The cluster environment already contained a manually edited NGRPO
implementation. The package `RECORD` hashes and the adjacent March 2026 backup
files confirmed that the edits were inside site-packages rather than upstream
ms-swift. The reroute keeps only the integration hooks in ms-swift and moves the
numerical operator into this repository.

## Integration Points

1. `swift/trainers/arguments.py`, `GRPOArgumentsMixin`

   Adds `ngrpo` to `advantage_estimator` and exposes the optional
   `ngrpo_virtual_max_reward` CLI field.

2. `swift/llm/argument/rlhf_args.py`, `GRPOArguments._init_grpo` and
   `GRPOArguments._check_grpo`

   Applies the GRPO defaults for `kl_in_reward` and `scale_rewards`, and permits
   NGRPO in the Liger validation path.

3. `swift/trainers/rlhf_trainer/grpo_trainer.py`,
   `GRPOTrainer._compute_advantages`, `compute_liger_loss`, and
   `_prepare_algorithm_params`

   Routes grouped rewards to the repository operator, accepts NGRPO for Liger
   loss preparation, and stores the configured virtual maximum reward.

## Commands

Check the interpreter without modifying it:

```bash
python scripts/patch_ms_swift_ngrpo.py
```

Apply the version-gated, idempotent patch:

```bash
python scripts/patch_ms_swift_ngrpo.py --apply
```

The script resolves the active `swift` installation instead of assuming a
site-packages path. It refuses versions other than 3.12.4 and creates
`*.pre_ngrpo_reroute` backups before the first write.
