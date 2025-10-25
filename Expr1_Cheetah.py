# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiment 1. Half-Cheetah
#

# %% [markdown]
# ## Setup
#

# %%
import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from rsl_rl.runners import OnPolicyRunner

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
env = registry.load("CheetahRun")
env

# %%
env_cfg = registry.get_default_config("CheetahRun")
env_cfg



rsl_cfg = registry.rsl_rl_config("CheetahRun")
rsl_cfg

# %% [markdown]
# ### Random Rollout Test
#

# %%
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(jax.random.PRNGKey(0))
rollout = [state]

frequency_hz = 0.5
for _ in range(env_cfg.episode_length):
    action = [
        jp.sin(
            state.data.time * 2 * jp.pi * frequency_hz + j * 2 * jp.pi / env.action_size
        )
        for j in range(env.action_size)
    ]
    state = jit_step(state, jp.array(action))
    rollout.append(state)

frames = env.render(rollout)
media.show_video(frames, fps=1.0 / env.dt)

# %% [markdown]
# ## Train
#

# %% [markdown]
# ### PPO
#

# %%
seed = 1
num_envs = 1024
num_steps_per_env = 32
max_iterations = 200

# %%
estimated_total_steps = max_iterations * num_envs * num_steps_per_env

brax_env = wrapper_torch.RSLRLBraxWrapper(
    raw_env,
    num_envs,
    _SEED.value,
    env_cfg.episode_length,
    1,
    render_callback=render_callback,
    randomization_fn=randomizer,
    device_rank=device_rank,
)
runner = OnPolicyRunner(brax_env, train_cfg_dict, device=device)

train_start = datetime.now()
runner.learn(max_iterations, init_at_random_ep_len=False)
train_end = datetime.now()


# %%

# %% [markdown]
# ### Option-Critic
#

# %%

# %% [markdown]
# ### Double Actor Critic
#

# %%

# %% [markdown]
# ## Comparison
#

# %% [markdown]
#
