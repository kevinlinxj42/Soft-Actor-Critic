from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.envs.registration import register


class SimpleContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self._state = np.zeros(1, dtype=np.float32)
        self._step = 0
        self._max_steps = 50

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step = 0
        self._state = self.np_random.uniform(low=-1.0, high=1.0, size=(1,)).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        self._step += 1
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._state = np.clip(self._state + 0.25 * action, -2.0, 2.0)
        reward = float(1.0 - abs(self._state[0]) - 0.05 * abs(action[0]))

        terminated = False
        truncated = self._step >= self._max_steps
        return self._state.copy(), reward, terminated, truncated, {}


@pytest.fixture(scope="session", autouse=True)
def register_test_env() -> None:
    try:
        register("SimpleContinuous-v0", entry_point=SimpleContinuousEnv)
    except gym.error.Error:
        # Already registered by another test worker/process.
        pass
