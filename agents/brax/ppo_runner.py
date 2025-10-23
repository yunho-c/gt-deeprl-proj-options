"""Brax PPO training helpers inspired by the MuJoCo Playground notebook."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional

import jax
from brax.training.agents import ppo
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jnp
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params

from envs import DMControlEnvConfig, apply_overrides, load_environment


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    """Container for outputs required to evaluate a trained PPO policy."""

    make_inference_fn: Callable[[Any, bool], Callable[[Any, Any], Any]]
    params: Any
    metrics: Dict[str, Iterable[float]]
    env_config: Any
    rl_config: Any
    total_time_s: float


def _build_progress_logger(
    progress_interval: int,
) -> Callable[[int, Dict[str, Any]], None]:
    """Creates a Brax progress_fn that logs metrics periodically."""

    if progress_interval <= 0:
        return lambda *_: None

    def progress(step_count: int, metrics: Dict[str, Any]) -> None:
        if step_count == 0:
            LOGGER.info("PPO compilation finished")
            return

        if step_count % progress_interval != 0:
            return

        reward = metrics.get("eval/episode_reward")
        reward_std = metrics.get("eval/episode_reward_std")
        LOGGER.info(
            "steps=%d eval_reward=%.3f±%.3f",
            step_count,
            float(reward) if reward is not None else float("nan"),
            float(reward_std) if reward_std is not None else float("nan"),
        )

    return progress


def train_dm_control_ppo(
    env_cfg: DMControlEnvConfig,
    rl_overrides: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 0,
    progress_interval: int = 1_000_000,
) -> TrainingArtifacts:
    """Trains a Brax PPO policy on a MuJoCo Playground DM Control environment."""

    env, resolved_env_cfg = load_environment(env_cfg)
    ppo_cfg = dm_control_suite_params.brax_ppo_config(env_cfg.env_name)

    if rl_overrides:
        apply_overrides(ppo_cfg, rl_overrides)

    training_kwargs = dict(ppo_cfg)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in training_kwargs:
        network_overrides = training_kwargs.pop("network_factory")
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **dict(network_overrides)
        )

    progress_fn = _build_progress_logger(progress_interval)
    train_fn = functools.partial(
        ppo.train,
        **training_kwargs,
        network_factory=network_factory,
        progress_fn=progress_fn,
        random_seed=seed,
    )

    start_time = perf_counter()
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    total_time = perf_counter() - start_time

    return TrainingArtifacts(
        make_inference_fn=make_inference_fn,
        params=params,
        metrics=metrics,
        env_config=resolved_env_cfg,
        rl_config=ppo_cfg,
        total_time_s=total_time,
    )


def rollout_policy(
    artifacts: TrainingArtifacts,
    *,
    env_cfg: DMControlEnvConfig,
    num_episodes: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    """Runs a greedy policy rollout to estimate episode returns."""

    env, resolved_env_cfg = load_environment(env_cfg)
    policy = jax.jit(artifacts.make_inference_fn(artifacts.params, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    episode_returns = []

    for _ in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        state = reset(reset_key)
        total_reward = 0.0
        for _ in range(resolved_env_cfg.episode_length):
            rng, action_key = jax.random.split(rng)
            action, _ = policy(state.obs, action_key)
            state = step(state, action)
            total_reward += float(jnp.asarray(state.reward))
        episode_returns.append(total_reward)

    avg_return = float(sum(episode_returns) / len(episode_returns))

    return {
        "returns": episode_returns,
        "avg_return": avg_return,
        "episode_length": resolved_env_cfg.episode_length,
    }


__all__ = [
    "TrainingArtifacts",
    "rollout_policy",
    "train_dm_control_ppo",
]
