"""CLI entry point to train Brax PPO agents on DM Control Suite tasks."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tomllib

from agents import rollout_policy, train_dm_control_ppo
from envs import DMControlEnvConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """User-specified training configuration loaded from TOML."""

    env_name: str
    seed: int = 0
    progress_interval: int = 1_000_000
    evaluation_episodes: int = 0
    env_overrides: Dict[str, Any] | None = None
    rl_overrides: Dict[str, Any] | None = None


def _load_config(path: Path) -> ExperimentConfig:
    raw = tomllib.loads(path.read_text())

    env_overrides = raw.get("env") or None
    rl_overrides = raw.get("rl") or None

    return ExperimentConfig(
        env_name=raw["env_name"],
        seed=raw.get("seed", 0),
        progress_interval=raw.get("progress_interval", 1_000_000),
        evaluation_episodes=raw.get("evaluation_episodes", 0),
        env_overrides=env_overrides,
        rl_overrides=rl_overrides,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mujoco/cartpole_balance.toml"),
        help="Path to the TOML experiment configuration.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    cfg = _load_config(args.config)
    env_cfg = DMControlEnvConfig(env_name=cfg.env_name, overrides=cfg.env_overrides)

    LOGGER.info("Starting PPO training for %s", cfg.env_name)
    artifacts = train_dm_control_ppo(
        env_cfg,
        cfg.rl_overrides,
        seed=cfg.seed,
        progress_interval=cfg.progress_interval,
    )
    LOGGER.info(
        "Training finished in %.2f s; final metrics keys=%s",
        artifacts.total_time_s,
        sorted(artifacts.metrics.keys()),
    )

    if cfg.evaluation_episodes > 0:
        eval_summary = rollout_policy(
            artifacts,
            env_cfg=env_cfg,
            num_episodes=cfg.evaluation_episodes,
            seed=cfg.seed + 1,
        )
        LOGGER.info(
            "Evaluation avg_return=%.3f over %d episodes",
            eval_summary["avg_return"],
            cfg.evaluation_episodes,
        )


if __name__ == "__main__":
    main()
