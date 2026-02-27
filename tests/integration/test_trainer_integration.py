from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sac.trainer import Trainer
from tests.helpers import build_test_config


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def test_training_smoke_no_nan(tmp_path: Path) -> None:
    cfg = build_test_config(output_dir=tmp_path, total_steps=80)
    run_dir = tmp_path / "smoke"
    trainer = Trainer(config=cfg, seed=7, run_dir=run_dir)
    trainer.train()

    assert trainer.global_step == 80

    train_metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    assert train_metrics_path.exists()
    assert eval_metrics_path.exists()

    rows = _load_jsonl(train_metrics_path)
    assert rows, "expected train metrics rows"

    for row in rows:
        for _, value in row.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value)


def test_checkpoint_resume_round_trip(tmp_path: Path) -> None:
    cfg = build_test_config(output_dir=tmp_path, total_steps=55)
    run_dir = tmp_path / "resume"
    trainer = Trainer(config=cfg, seed=11, run_dir=run_dir)
    trainer.train()

    ckpt_path = run_dir / "checkpoint_step_55.pt"
    assert ckpt_path.exists()

    resumed_cfg = build_test_config(output_dir=tmp_path, total_steps=70)
    trainer2 = Trainer(config=resumed_cfg, seed=11, run_dir=tmp_path / "resume_2")
    trainer2.resume(ckpt_path)

    assert trainer2.global_step == 55

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expected_actor_optim = checkpoint["agent"]["actor_optim"]
    loaded_actor_optim = trainer2.agent.actor_optim.state_dict()
    assert expected_actor_optim["param_groups"] == loaded_actor_optim["param_groups"]

    trainer2.train()
    assert trainer2.global_step == 70


def test_deterministic_eval_repeatable(tmp_path: Path) -> None:
    cfg = build_test_config(output_dir=tmp_path, total_steps=20)
    run_dir = tmp_path / "eval_repeatable"
    trainer = Trainer(config=cfg, seed=1234, run_dir=run_dir)

    metrics1 = trainer.evaluate(num_episodes=4)
    metrics2 = trainer.evaluate(num_episodes=4)

    assert metrics1 == metrics2


def test_corrupted_checkpoint_raises(tmp_path: Path) -> None:
    cfg = build_test_config(output_dir=tmp_path, total_steps=20)
    run_dir = tmp_path / "bad_ckpt"
    trainer = Trainer(config=cfg, seed=42, run_dir=run_dir)

    bad_ckpt = tmp_path / "corrupt.pt"
    bad_ckpt.write_text("not a checkpoint", encoding="utf-8")

    with pytest.raises(Exception):
        trainer.resume(bad_ckpt)
