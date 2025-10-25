# Helper utilities for configuring and running RSL-RL training inside the notebook.

import math
import os
import statistics
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output, display
from ml_collections import config_dict
from mujoco_playground import registry, wrapper_torch
from rsl_rl.runners import OnPolicyRunner


def dm_control_rsl_config(
    env_name: str,
    *,
    num_envs: int,
    num_steps_per_env: int,
    max_iterations: int,
    seed: int,
) -> config_dict.ConfigDict:
    """Constructs a lightweight PPO config for DM Control tasks."""
    cfg = config_dict.create(
        seed=seed,
        runner_class_name="OnPolicyRunner",
        policy=config_dict.create(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            activation="elu",
            class_name="ActorCritic",
        ),
        algorithm=config_dict.create(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        empirical_normalization=True,
        obs_groups={},
        save_interval=max(50, max_iterations // 5),
        experiment_name=f"{env_name}_rsl",
        run_name="",
        resume=False,
        load_run="-1",
        checkpoint=-1,
        resume_path=None,
        logger="tensorboard",
    )

    cfg.num_envs = num_envs

    # Slight tweaks for particularly challenging tasks.
    if env_name in ("CheetahRun", "WalkerRun", "HumanoidRun"):
        cfg.algorithm.entropy_coef = 0.01
        cfg.algorithm.num_learning_epochs = 5
        cfg.algorithm.num_mini_batches = 8
        cfg.policy.actor_hidden_dims = [256, 256, 128]
        cfg.policy.critic_hidden_dims = [256, 256, 128]

    return cfg


class NotebookOnPolicyRunner(OnPolicyRunner):
    """Subclass of the RSL-RL runner that captures metrics for notebook plotting."""

    def __init__(
        self,
        env,
        train_cfg: Dict,
        log_dir: str,
        device: str = "cuda:0",
        progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ):
        self._progress_callback = progress_callback
        self.progress_history: List[Dict[str, float]] = []
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def _prepare_logging_writer(self):
        if self.log_dir is not None and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.writer = None
        self.logger_type = "notebook"

    def log(self, locs: Dict, width: int = 80, pad: int = 35):  # pylint: disable=unused-argument
        collection_size = (
            self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        )
        iteration_time = locs["collection_time"] + locs["learn_time"]

        self.tot_timesteps += collection_size
        self.tot_time += iteration_time

        reward_mean = (
            statistics.mean(locs["rewbuffer"]) if locs["rewbuffer"] else float("nan")
        )
        reward_std = (
            statistics.pstdev(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 1 else 0.0
        )
        episode_length = (
            statistics.mean(locs["lenbuffer"]) if locs["lenbuffer"] else float("nan")
        )
        fps = collection_size / iteration_time if iteration_time > 0 else float("nan")
        action_std = self.alg.policy.action_std.mean().item()

        losses = {
            key: value.item() if hasattr(value, "item") else float(value)
            for key, value in locs["loss_dict"].items()
        }

        metrics = {
            "iteration": locs["it"],
            "total_iterations": locs["tot_iter"],
            "total_timesteps": self.tot_timesteps,
            "fps": fps,
            "collection_time": locs["collection_time"],
            "learn_time": locs["learn_time"],
            "episode_reward_mean": reward_mean,
            "episode_reward_std": reward_std,
            "episode_length_mean": episode_length,
            "action_std": action_std,
            "losses": losses,
        }

        self.progress_history.append(metrics)

        if self._progress_callback is not None and locs["rewbuffer"]:
            self._progress_callback(self.tot_timesteps, metrics)


def make_progress_plotter(expected_steps: int):
    """Creates a Matplotlib-backed progress callback."""
    steps: List[int] = []
    rewards: List[float] = []
    reward_stds: List[float] = []
    timestamps = [datetime.now()]

    def _callback(total_steps: int, metrics: Dict[str, float]):
        reward = metrics["episode_reward_mean"]
        if not math.isfinite(reward):
            return

        std = metrics["episode_reward_std"]
        if not math.isfinite(std):
            std = 0.0

        steps.append(int(total_steps))
        rewards.append(float(reward))
        reward_stds.append(float(std))
        timestamps.append(datetime.now())

        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(steps, rewards, yerr=reward_stds, color="blue")
        ax.set_xlabel("# environment steps")
        ax.set_ylabel("episode reward")
        ax.set_xlim([0, max(expected_steps * 1.05, steps[-1] * 1.05)])
        ax.set_title(
            f"iter {metrics['iteration']} / {metrics['total_iterations']}  "
            f"| reward {rewards[-1]:.2f}"
        )
        ax.grid(True, linestyle="--", linewidth=0.3)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    return _callback, steps, rewards, reward_stds, timestamps


def setup_rsl_runner(
    env_name: str,
    *,
    seed: int,
    num_envs: int,
    num_steps_per_env: int,
    max_iterations: int,
    progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    device: Optional[str] = None,
) -> Tuple[NotebookOnPolicyRunner, config_dict.ConfigDict, str, str]:
    """Initialises the environment wrapper and RSL-RL runner."""
    env_cfg = registry.get_default_config(env_name)
    raw_env = registry.load(env_name, config=env_cfg)
    randomizer = registry.get_domain_randomizer(env_name)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device {device}, but PyTorch reports no GPU."
        )

    device_rank = None
    if device.startswith("cuda"):
        device_rank = int(device.split(":")[-1])

    train_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        seed,
        env_cfg.episode_length,
        action_repeat=1,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )
    if device_rank is None:
        train_env.device = device

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", "dm_control_rsl", f"{env_name}-{timestamp}")
    os.makedirs(logdir, exist_ok=True)

    train_cfg = dm_control_rsl_config(
        env_name,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        seed=seed,
    )
    train_cfg.run_name = f"{env_name}_seed{seed}"
    train_cfg.experiment_name = train_cfg.run_name
    train_cfg.save_interval = max(10, max_iterations // 5)

    train_cfg_dict = train_cfg.to_dict()
    train_cfg_dict["seed"] = seed

    runner = NotebookOnPolicyRunner(
        train_env,
        train_cfg_dict,
        log_dir=logdir,
        device=device,
        progress_callback=progress_callback,
    )

    return runner, env_cfg, logdir, device


last_runner: Optional[NotebookOnPolicyRunner] = None
last_env_cfg: Optional[config_dict.ConfigDict] = None
last_env_name: Optional[str] = None
last_device: Optional[str] = None
last_progress: Dict[str, List] = {}
