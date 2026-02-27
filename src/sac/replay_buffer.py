from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class ReplayBatch:
    obs: Tensor
    act: Tensor
    rew: Tensor
    next_obs: Tensor
    done: Tensor


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype

        self.obs = np.zeros((self.capacity, obs_dim), dtype=self.dtype)
        self.actions = np.zeros((self.capacity, action_dim), dtype=self.dtype)
        self.rewards = np.zeros((self.capacity, 1), dtype=self.dtype)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=self.dtype)
        self.dones = np.zeros((self.capacity, 1), dtype=self.dtype)

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def pos(self) -> int:
        return self._pos

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        idx = self._pos
        self.obs[idx] = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.next_obs[idx] = next_obs
        self.dones[idx] = done

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self._size < batch_size:
            raise ValueError(f"Cannot sample batch_size={batch_size} from size={self._size}")

        idxs = np.random.randint(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[idxs], dtype=torch.float32, device=self.device),
            act=torch.as_tensor(self.actions[idxs], dtype=torch.float32, device=self.device),
            rew=torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idxs], dtype=torch.float32, device=self.device),
            done=torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=self.device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "pos": self._pos,
            "size": self._size,
            "obs": self.obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_obs": self.next_obs,
            "dones": self.dones,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        capacity = int(state_dict["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: ckpt={capacity}, current={self.capacity}. "
                "Initialize with matching replay_size to resume."
            )

        self._pos = int(state_dict["pos"])
        self._size = int(state_dict["size"])
        self.obs[...] = state_dict["obs"]
        self.actions[...] = state_dict["actions"]
        self.rewards[...] = state_dict["rewards"]
        self.next_obs[...] = state_dict["next_obs"]
        self.dones[...] = state_dict["dones"]
