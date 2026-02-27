from __future__ import annotations

import torch

from sac.networks import GaussianActor, tanh_normal_log_prob


def test_tanh_normal_log_prob_matches_manual_formula() -> None:
    torch.manual_seed(0)
    raw_action = torch.randn(4, 3)
    mean = torch.randn(4, 3)
    log_std = torch.randn(4, 3).clamp(-5, 2)

    got = tanh_normal_log_prob(raw_action, mean, log_std)

    std = torch.exp(log_std)
    dist = torch.distributions.Normal(mean, std)
    y = torch.tanh(raw_action)
    expected = (dist.log_prob(raw_action) - torch.log(1.0 - y.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)

    assert torch.allclose(got, expected, atol=1e-6)


def test_actor_sample_shapes_and_finite_values() -> None:
    torch.manual_seed(0)
    actor = GaussianActor(obs_dim=5, action_dim=2, hidden_dims=(32, 32))
    obs = torch.randn(7, 5)

    output = actor.sample(obs)

    assert output.action.shape == (7, 2)
    assert output.log_prob.shape == (7, 1)
    assert output.mean_action.shape == (7, 2)
    assert torch.isfinite(output.action).all()
    assert torch.isfinite(output.log_prob).all()
    assert torch.max(torch.abs(output.action)) <= 1.0
