from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def tanh_normal_log_prob(raw_action: Tensor, mean: Tensor, log_std: Tensor, eps: float = 1e-6) -> Tensor:
    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    squashed_action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed_action.pow(2) + eps)
    return log_prob.sum(dim=-1, keepdim=True)


class PolicyOutput:
    def __init__(self, action: Tensor, log_prob: Tensor, mean_action: Tensor) -> None:
        self.action = action
        self.log_prob = log_prob
        self.mean_action = mean_action


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mu, log_std

    def sample(self, obs: Tensor) -> PolicyOutput:
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = tanh_normal_log_prob(x_t, mu, log_std)

        mean_action = torch.tanh(mu)
        return PolicyOutput(action=y_t, log_prob=log_prob, mean_action=mean_action)

    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        if deterministic:
            mu, _ = self.forward(obs)
            return torch.tanh(mu)
        return self.sample(obs).action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.q1_net = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2_net = QNetwork(obs_dim, action_dim, hidden_dims)

    def q1(self, obs: Tensor, act: Tensor) -> Tensor:
        return self.q1_net(obs, act)

    def q2(self, obs: Tensor, act: Tensor) -> Tensor:
        return self.q2_net(obs, act)

    def forward(self, obs: Tensor, act: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(obs, act), self.q2(obs, act)
