from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import pytest

from sac.config import EnvConfig, ExperimentConfig, SACConfig, TrainConfig
from sac.trainer import Trainer
from tests.helpers import build_test_config


def _load_last_eval_return(run_dir: Path) -> float:
    eval_path = run_dir / "eval_metrics.jsonl"
    with eval_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return float(rows[-1]["eval_return_mean"])


def test_smoke_performance_floor(tmp_path: Path) -> None:
    cfg = build_test_config(output_dir=tmp_path, total_steps=120)
    run_dir = tmp_path / "perf_floor"

    trainer = Trainer(config=cfg, seed=101, run_dir=run_dir)
    trainer.train()

    final_eval = _load_last_eval_return(run_dir)
    assert final_eval > -2.0


def test_invalid_env_id_fails_fast(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        sac=SACConfig(),
        env=EnvConfig(env_id="ThisEnvDoesNotExist-v0"),
        train=TrainConfig(total_steps=10, seed_list=[0], output_dir=str(tmp_path), device="cpu"),
    )

    with pytest.raises(gym.error.Error):
        Trainer(config=cfg, seed=0, run_dir=tmp_path / "bad_env")
