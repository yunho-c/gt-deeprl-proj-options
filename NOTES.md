# Project Structure Notes

## Proposed Layout
- `pyproject.toml`: single-source environment spec with tasks like `pixi run train`, `eval`, and `lint`.
- `configs/`: structured YAML/TOML configs for Mujoco assets, training hyperparams, and logging presets.
- `envs/`: environment wrappers plus `mujoco_playground/` assets/utilities; export gym-style registration helpers.
- `agents/`: `rsl_rl`-based policies, networks, rollout storage, and algorithm-specific modules.
- `experiments/`: CLI entry points, shell/Pixi task scripts, and per-run settings; keep notebooks here tied to runs.
- `artifacts/`: gitignored outputs—checkpoints, evaluation metrics, TensorBoard logs, replay buffers.

```
gt-deeprl-proj-options/
├── pyproject.toml
├── configs/
│   ├── mujoco/
│   └── agents/
├── envs/
│   ├── mujoco_playground/
│   ├── wrappers/
│   └── benchmarks.py
├── agents/
│   ├── common/
│   ├── ppo/
│   └── options/
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── 2024-06-15_ant_ppo/
└── artifacts/  (gitignored)
```

## Details
- Mirror algorithm variants under `agents/<algo>/` (e.g., `ppo`, `ppo_options`) with shared utilities in `agents/common/`.
- Split `envs/` into `mujoco_playground/`, `wrappers/` (reward shaping, normalization), and `benchmarks/` for task registries.
- Model reproducible runs as `experiments/<date>_<task>_<agent>/` with symlinks into `artifacts/` for checkpoints and summaries.
- Use Pixi tasks to orchestrate workflows: `train` launches `python experiments/train.py`, `eval` reports metrics, `plot` executes notebooks headlessly.

## Implemented Files
- `envs/dm_control.py`: loads MuJoCo Playground DM Control Suite envs and applies nested overrides for quick experimentation.
- `agents/brax/ppo_runner.py`: wraps the notebook's PPO training loop in a reusable function that logs progress and exposes rollouts.
- `experiments/train.py`: CLI driver that reads a TOML config, spins up the environment, runs Brax PPO, and optionally evaluates the trained policy.
- `configs/mujoco/cartpole_balance.toml`: sample config mirroring the notebook setup with smaller defaults for fast local sanity checks.
