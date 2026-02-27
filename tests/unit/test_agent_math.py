from __future__ import annotations

import types

import torch

from sac.agent import SACAgent
from sac.config import SACConfig
from sac.replay_buffer import ReplayBatch


def _make_agent(*, use_twin_q: bool = True, tau: float = 0.5) -> SACAgent:
    config = SACConfig(hidden_dims=(32, 32), batch_size=2, use_twin_q=use_twin_q, tau=tau)
    return SACAgent(obs_dim=3, action_dim=1, config=config, device=torch.device("cpu"))


def test_target_q_formula_with_done_mask() -> None:
    agent = _make_agent(use_twin_q=True)

    batch = ReplayBatch(
        obs=torch.zeros((2, 3)),
        act=torch.zeros((2, 1)),
        rew=torch.tensor([[1.0], [2.0]]),
        next_obs=torch.zeros((2, 3)),
        done=torch.tensor([[0.0], [1.0]]),
    )

    fake_policy = types.SimpleNamespace(
        action=torch.zeros((2, 1)),
        log_prob=torch.full((2, 1), 0.5),
    )
    agent.actor.sample = lambda obs: fake_policy  # type: ignore[method-assign]

    def fake_forward(self, obs, act):
        return torch.full((2, 1), 2.0), torch.full((2, 1), 3.0)

    agent.target_critic.forward = types.MethodType(fake_forward, agent.target_critic)

    target_q, _ = agent._compute_target_q(batch)

    expected_bootstrap = 2.0 - agent.alpha.item() * 0.5
    expected = torch.tensor(
        [
            [1.0 + agent.config.gamma * expected_bootstrap],
            [2.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(target_q, expected, atol=1e-5)


def test_soft_update_polyak_average() -> None:
    agent = _make_agent(tau=0.25)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.fill_(1.0)
        for p in agent.target_critic.parameters():
            p.zero_()

    agent._soft_update_targets()

    with torch.no_grad():
        for p in agent.target_critic.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.25), atol=1e-6)


def test_alpha_update_directionality() -> None:
    agent = _make_agent()

    alpha_before = agent.alpha.item()
    low_entropy_log_prob = torch.full((64, 1), 2.0)
    alpha_loss_up = -(agent.log_alpha * (low_entropy_log_prob + agent.target_entropy)).mean()
    agent.alpha_optim.zero_grad(set_to_none=True)
    alpha_loss_up.backward()
    agent.alpha_optim.step()

    alpha_after_up = agent.alpha.item()
    assert alpha_after_up > alpha_before

    high_entropy_log_prob = torch.full((64, 1), -10.0)
    alpha_loss_down = -(agent.log_alpha * (high_entropy_log_prob + agent.target_entropy)).mean()
    agent.alpha_optim.zero_grad(set_to_none=True)
    alpha_loss_down.backward()
    agent.alpha_optim.step()

    alpha_after_down = agent.alpha.item()
    assert alpha_after_down < alpha_after_up
