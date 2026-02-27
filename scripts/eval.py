#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sac.config import load_config
from sac.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--seed", type=int, default=0, help="Seed used to build eval env")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    eval_dir = Path(config.train.output_dir) / "eval"
    trainer = Trainer(config=config, seed=args.seed, run_dir=eval_dir)
    trainer.resume(args.checkpoint)

    metrics = trainer.evaluate(num_episodes=args.episodes)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
