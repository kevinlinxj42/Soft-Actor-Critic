from __future__ import annotations

from pathlib import Path

from sac.config import EnvConfig, ExperimentConfig, SACConfig, TrainConfig


def build_test_config(output_dir: str | Path, total_steps: int = 120) -> ExperimentConfig:
    cfg = ExperimentConfig(
        sac=SACConfig(
            gamma=0.99,
            tau=0.01,
            batch_size=32,
            replay_size=5000,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            hidden_dims=(64, 64),
            warmup_steps=20,
            updates_per_step=1,
            target_update_interval=1,
            use_twin_q=True,
        ),
        env=EnvConfig(
            env_id="SimpleContinuous-v0",
            max_episode_steps=50,
        ),
        train=TrainConfig(
            total_steps=total_steps,
            eval_interval=40,
            eval_episodes=4,
            checkpoint_interval=50,
            log_interval=10,
            seed_list=[0],
            device="cpu",
            deterministic_torch=True,
            run_name="tests",
            output_dir=str(output_dir),
            save_replay_in_checkpoint=True,
        ),
    )
    cfg.validate()
    return cfg
