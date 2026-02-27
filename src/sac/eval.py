from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sac.envs import ActionBounds, scale_action


@dataclass
class EvalResult:
    mean_return: float
    std_return: float
    mean_episode_length: float


def evaluate_policy(
    agent,
    env,
    action_bounds: ActionBounds,
    num_episodes: int,
    seed: int | None = None,
) -> EvalResult:
    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(num_episodes):
        reset_seed = None if seed is None else seed + ep
        obs, _ = env.reset(seed=reset_seed)
        done = False
        truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (done or truncated):
            action_norm = agent.select_action(obs, eval_mode=True)
            action = scale_action(action_norm, action_bounds)
            obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    return EvalResult(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_episode_length=float(np.mean(lengths)),
    )
