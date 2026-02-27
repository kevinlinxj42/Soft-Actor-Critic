from __future__ import annotations

import numpy as np
import torch

from sac.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_shapes_and_dtypes() -> None:
    device = torch.device("cpu")
    buf = ReplayBuffer(obs_dim=3, action_dim=2, capacity=128, device=device)

    for i in range(64):
        obs = np.ones(3, dtype=np.float32) * i
        act = np.ones(2, dtype=np.float32) * (i / 10.0)
        rew = float(i)
        next_obs = obs + 1
        done = float(i % 2)
        buf.add(obs, act, rew, next_obs, done)

    batch = buf.sample(16)
    assert batch.obs.shape == (16, 3)
    assert batch.act.shape == (16, 2)
    assert batch.rew.shape == (16, 1)
    assert batch.next_obs.shape == (16, 3)
    assert batch.done.shape == (16, 1)

    assert batch.obs.dtype == torch.float32
    assert batch.act.dtype == torch.float32
    assert batch.obs.device.type == "cpu"


def test_replay_state_dict_round_trip() -> None:
    device = torch.device("cpu")
    src = ReplayBuffer(obs_dim=2, action_dim=1, capacity=32, device=device)
    for i in range(20):
        src.add(
            obs=np.array([i, i + 0.5], dtype=np.float32),
            act=np.array([i * 0.1], dtype=np.float32),
            rew=float(i),
            next_obs=np.array([i + 1, i + 1.5], dtype=np.float32),
            done=float(i % 3 == 0),
        )

    dst = ReplayBuffer(obs_dim=2, action_dim=1, capacity=32, device=device)
    dst.load_state_dict(src.state_dict())

    assert len(src) == len(dst)
    assert src.pos == dst.pos
    assert np.allclose(src.obs, dst.obs)
    assert np.allclose(src.actions, dst.actions)
    assert np.allclose(src.rewards, dst.rewards)
    assert np.allclose(src.next_obs, dst.next_obs)
    assert np.allclose(src.dones, dst.dones)
