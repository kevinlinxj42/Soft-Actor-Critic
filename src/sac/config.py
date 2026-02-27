from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 1_000_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float | None = None
    hidden_dims: tuple[int, int] = (256, 256)
    warmup_steps: int = 10_000
    updates_per_step: int = 1
    target_update_interval: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    use_twin_q: bool = True

    def validate(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.replay_size < self.batch_size:
            raise ValueError("replay_size must be >= batch_size")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.alpha_lr <= 0:
            raise ValueError("all learning rates must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.updates_per_step <= 0:
            raise ValueError("updates_per_step must be > 0")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be > 0")
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")


@dataclass
class EnvConfig:
    env_id: str = "HalfCheetah-v4"
    max_episode_steps: int | None = None
    action_clip: bool = True
    train_seed_offset: int = 0
    eval_seed_offset: int = 10_000

    def validate(self) -> None:
        if not self.env_id:
            raise ValueError("env_id must be a non-empty string")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0 when provided")


@dataclass
class TrainConfig:
    total_steps: int = 1_000_000
    eval_interval: int = 5_000
    eval_episodes: int = 10
    checkpoint_interval: int = 50_000
    log_interval: int = 1_000
    seed_list: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "cuda"
    deterministic_torch: bool = False
    run_name: str = "sac"
    output_dir: str = "runs"
    save_replay_in_checkpoint: bool = False

    def validate(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be > 0")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be > 0")
        if not self.seed_list:
            raise ValueError("seed_list cannot be empty")


@dataclass
class ExperimentConfig:
    sac: SACConfig = field(default_factory=SACConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.sac.validate()
        self.env.validate()
        self.train.validate()


def _merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def _to_hidden_dims(value: Any) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    raise ValueError(f"hidden_dims must be a list or tuple, got {type(value)}")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    sac_raw = raw.get("sac", {})
    env_raw = raw.get("env", {})
    train_raw = raw.get("train", {})

    if "hidden_dims" in sac_raw:
        sac_raw = dict(sac_raw)
        sac_raw["hidden_dims"] = _to_hidden_dims(sac_raw["hidden_dims"])

    config = ExperimentConfig(
        sac=SACConfig(**sac_raw),
        env=EnvConfig(**env_raw),
        train=TrainConfig(**train_raw),
    )
    config.validate()
    return config


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML root must be a mapping")
    if overrides:
        _merge_dict(raw, overrides)
    return _build_config(raw)
