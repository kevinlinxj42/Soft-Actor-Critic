from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics

from sac.config import EnvConfig


@dataclass
class ActionBounds:
    low: np.ndarray
    high: np.ndarray


def set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic_torch
    torch.backends.cudnn.benchmark = not deterministic_torch


def make_env(config: EnvConfig, seed: int, eval_mode: bool = False) -> gym.Env:
    kwargs: dict[str, int] = {}
    if config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.max_episode_steps

    env = gym.make(config.env_id, **kwargs)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed + (config.eval_seed_offset if eval_mode else config.train_seed_offset))

    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("SAC implementation requires a continuous Box action space")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("SAC implementation requires a Box observation space")
    return env


def action_bounds_from_env(env: gym.Env) -> ActionBounds:
    assert isinstance(env.action_space, gym.spaces.Box)
    return ActionBounds(low=env.action_space.low.astype(np.float32), high=env.action_space.high.astype(np.float32))


def scale_action(action: np.ndarray, bounds: ActionBounds) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return bounds.low + 0.5 * (action + 1.0) * (bounds.high - bounds.low)


def unscale_action(action: np.ndarray, bounds: ActionBounds) -> np.ndarray:
    scaled = 2.0 * (action - bounds.low) / (bounds.high - bounds.low + 1e-8) - 1.0
    return np.clip(scaled, -1.0, 1.0)
