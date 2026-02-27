from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sac.agent import SACAgent
from sac.config import ExperimentConfig
from sac.envs import action_bounds_from_env, make_env, scale_action, set_global_seeds
from sac.eval import evaluate_policy
from sac.logging import MetricLogger
from sac.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, config: ExperimentConfig, seed: int, run_dir: str | Path) -> None:
        self.config = config
        self.seed = int(seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        set_global_seeds(self.seed, deterministic_torch=config.train.deterministic_torch)

        use_cuda = config.train.device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(config.train.device if use_cuda else "cpu")

        self.env = make_env(config.env, seed=self.seed, eval_mode=False)
        self.eval_env = make_env(config.env, seed=self.seed, eval_mode=True)

        assert hasattr(self.env.observation_space, "shape")
        assert hasattr(self.env.action_space, "shape")
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = int(np.prod(self.env.action_space.shape))

        self.action_bounds = action_bounds_from_env(self.env)

        self.agent = SACAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=config.sac,
            device=self.device,
        )
        self.replay = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=config.sac.replay_size,
            device=self.device,
        )

        self.logger = MetricLogger(self.run_dir)
        self.logger.write_metadata(
            {
                "seed": self.seed,
                "env_id": config.env.env_id,
                "device": str(self.device),
                "sac": config.sac.__dict__,
                "train": config.train.__dict__,
            }
        )

        self.global_step = 0
        self.episode_idx = 0
        self._obs, _ = self.env.reset(seed=self.seed + config.env.train_seed_offset)
        self._episode_return = 0.0
        self._episode_length = 0
        self._start_time = time.perf_counter()

    def _checkpoint_path(self, step: int) -> Path:
        return self.run_dir / f"checkpoint_step_{step}.pt"

    def _collect_transition(self, action_norm: np.ndarray) -> tuple[float, bool, dict[str, Any]]:
        action = scale_action(action_norm, self.action_bounds)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        done = bool(terminated or truncated)
        done_for_bootstrap = float(terminated)

        if not np.isfinite(reward):
            raise FloatingPointError("Encountered non-finite reward")
        if not np.all(np.isfinite(next_obs)):
            raise FloatingPointError("Encountered non-finite observation")

        self.replay.add(self._obs, action_norm, float(reward), next_obs, done_for_bootstrap)

        self._obs = next_obs
        self._episode_return += float(reward)
        self._episode_length += 1

        if done:
            self.episode_idx += 1
            episode_metrics = {
                "episode_return": float(self._episode_return),
                "episode_length": float(self._episode_length),
                "episode_idx": float(self.episode_idx),
            }
            self._obs, _ = self.env.reset()
            self._episode_return = 0.0
            self._episode_length = 0
        else:
            episode_metrics = {
                "episode_return": float("nan"),
                "episode_length": float("nan"),
                "episode_idx": float("nan"),
            }

        return float(reward), done, episode_metrics

    def _sample_action(self) -> np.ndarray:
        if self.global_step < self.config.sac.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
        return self.agent.select_action(self._obs, eval_mode=False).astype(np.float32)

    def _update_agent(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "critic_loss": float("nan"),
            "critic_loss_q1": float("nan"),
            "critic_loss_q2": float("nan"),
            "actor_loss": float("nan"),
            "alpha_loss": float("nan"),
            "alpha": float("nan"),
            "target_q_mean": float("nan"),
            "policy_entropy": float("nan"),
        }
        if len(self.replay) < self.config.sac.batch_size:
            return metrics
        if self.global_step < self.config.sac.warmup_steps:
            return metrics

        for _ in range(self.config.sac.updates_per_step):
            batch = self.replay.sample(self.config.sac.batch_size)
            metrics.update(self.agent.update(batch))
        return metrics

    def _train_log_payload(
        self,
        reward: float,
        update_metrics: dict[str, float],
        episode_metrics: dict[str, float],
    ) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        steps_per_sec = self.global_step / elapsed
        payload: dict[str, float] = {
            "reward": reward,
            "replay_size": float(len(self.replay)),
            "replay_fill_ratio": float(len(self.replay) / self.config.sac.replay_size),
            "steps_per_sec": float(steps_per_sec),
        }
        payload.update(update_metrics)
        payload.update(episode_metrics)
        return payload

    def evaluate(self, num_episodes: int | None = None) -> dict[str, float]:
        episodes = num_episodes or self.config.train.eval_episodes
        result = evaluate_policy(
            self.agent,
            self.eval_env,
            self.action_bounds,
            episodes,
            seed=self.seed + self.config.env.eval_seed_offset,
        )
        metrics = {
            "eval_return_mean": result.mean_return,
            "eval_return_std": result.std_return,
            "eval_episode_length_mean": result.mean_episode_length,
        }
        self.logger.log_eval(self.global_step, metrics)
        return metrics

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path else self._checkpoint_path(self.global_step)

        checkpoint: dict[str, Any] = {
            "global_step": self.global_step,
            "episode_idx": self.episode_idx,
            "obs": self._obs,
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
            "agent": self.agent.state_dict(),
            "replay_meta": {
                "capacity": self.replay.capacity,
                "size": len(self.replay),
                "pos": self.replay.pos,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if self.config.train.save_replay_in_checkpoint:
            checkpoint["replay"] = self.replay.state_dict()

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.global_step = int(checkpoint["global_step"])
        self.episode_idx = int(checkpoint.get("episode_idx", 0))
        self._obs = np.asarray(checkpoint.get("obs", self._obs), dtype=np.float32)
        self._episode_return = float(checkpoint.get("episode_return", 0.0))
        self._episode_length = int(checkpoint.get("episode_length", 0))

        self.agent.load_state_dict(checkpoint["agent"])

        if "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])

        rng_state = checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

        self._start_time = time.perf_counter()

    def train(self) -> None:
        while self.global_step < self.config.train.total_steps:
            action_norm = self._sample_action()
            reward, _, episode_metrics = self._collect_transition(action_norm)
            self.global_step += 1

            update_metrics = self._update_agent()

            if self.global_step % self.config.train.log_interval == 0:
                train_metrics = self._train_log_payload(reward, update_metrics, episode_metrics)
                self.logger.log_train(self.global_step, train_metrics)

            if self.global_step % self.config.train.eval_interval == 0:
                self.evaluate(self.config.train.eval_episodes)

            if self.global_step % self.config.train.checkpoint_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint(self._checkpoint_path(self.global_step))
