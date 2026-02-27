from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from sac.config import SACConfig
from sac.networks import DoubleQCritic, GaussianActor
from sac.replay_buffer import ReplayBatch


def _to_tensor(obs: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: SACConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.action_dim = action_dim

        self.actor = GaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
        ).to(self.device)
        self.critic = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=config.hidden_dims).to(self.device)
        self.target_critic = DoubleQCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=config.hidden_dims).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        initial_log_alpha = torch.log(torch.tensor(config.init_alpha, dtype=torch.float32, device=self.device))
        self.log_alpha = nn.Parameter(initial_log_alpha)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        if config.target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(config.target_entropy)

        self.train_step = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = _to_tensor(obs, self.device)
            action = self.actor.act(obs_t, deterministic=eval_mode)
            action_np = action.squeeze(0).cpu().numpy()
        return np.clip(action_np, -1.0, 1.0)

    def _compute_target_q(self, batch: ReplayBatch) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            next_policy = self.actor.sample(batch.next_obs)
            target_q1, target_q2 = self.target_critic(batch.next_obs, next_policy.action)
            if self.config.use_twin_q:
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = target_q1
            target_v = target_q - self.alpha.detach() * next_policy.log_prob
            target = batch.rew + (1.0 - batch.done) * self.config.gamma * target_v
        return target, next_policy.log_prob

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        self.train_step += 1

        target_q, next_log_prob = self._compute_target_q(batch)
        q1_pred, q2_pred = self.critic(batch.obs, batch.act)

        critic_loss_q1 = torch.nn.functional.mse_loss(q1_pred, target_q)
        if self.config.use_twin_q:
            critic_loss_q2 = torch.nn.functional.mse_loss(q2_pred, target_q)
            critic_loss = critic_loss_q1 + critic_loss_q2
        else:
            critic_loss_q2 = torch.zeros_like(critic_loss_q1)
            critic_loss = critic_loss_q1

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optim.step()

        policy = self.actor.sample(batch.obs)
        q1_pi, q2_pi = self.critic(batch.obs, policy.action)
        q_pi = torch.min(q1_pi, q2_pi) if self.config.use_twin_q else q1_pi

        actor_loss = (self.alpha.detach() * policy.log_prob - q_pi).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optim.step()

        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (policy.log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        if self.train_step % self.config.target_update_interval == 0:
            self._soft_update_targets()

        for metric_name, value in {
            "critic_loss": critic_loss,
            "critic_loss_q1": critic_loss_q1,
            "critic_loss_q2": critic_loss_q2,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
            "target_q_mean": target_q.mean(),
            "policy_entropy": (-policy.log_prob).mean(),
            "next_log_prob_mean": next_log_prob.mean(),
        }.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise FloatingPointError(f"{metric_name} became invalid during update")

        return {
            "critic_loss": float(critic_loss.item()),
            "critic_loss_q1": float(critic_loss_q1.item()),
            "critic_loss_q2": float(critic_loss_q2.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "target_q_mean": float(target_q.mean().item()),
            "policy_entropy": float((-policy.log_prob).mean().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "train_step": self.train_step,
            "target_entropy": self.target_entropy,
            "config": {
                "gamma": self.config.gamma,
                "tau": self.config.tau,
                "batch_size": self.config.batch_size,
                "use_twin_q": self.config.use_twin_q,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])

        loaded_log_alpha = state_dict["log_alpha"]
        if isinstance(loaded_log_alpha, torch.Tensor):
            self.log_alpha.data.copy_(loaded_log_alpha.to(self.device))
        else:
            self.log_alpha.data.copy_(torch.tensor(loaded_log_alpha, device=self.device))
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        self.train_step = int(state_dict.get("train_step", 0))
        self.target_entropy = float(state_dict.get("target_entropy", self.target_entropy))

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint)
