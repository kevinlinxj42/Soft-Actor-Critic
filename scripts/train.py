#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from sac.config import load_config
from sac.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on Gymnasium environments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list override. Defaults to train.seed_list from config.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional suffix for run directory naming",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = config.train.seed_list
    if args.resume and len(seeds) != 1:
        raise ValueError("--resume requires exactly one target seed")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""

    for seed in seeds:
        run_dir = (
            Path(config.train.output_dir)
            / config.train.run_name
            / config.env.env_id
            / f"seed_{seed}_{ts}{run_suffix}"
        )
        trainer = Trainer(config=config, seed=seed, run_dir=run_dir)

        if args.resume:
            trainer.resume(args.resume)

        print(f"[train] seed={seed} run_dir={run_dir}")
        trainer.train()
        print(f"[done] seed={seed} final_step={trainer.global_step}")


if __name__ == "__main__":
    main()
