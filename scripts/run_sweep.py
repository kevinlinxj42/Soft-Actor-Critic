#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sac.config import load_config
from sac.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAC ablation sweeps")
    parser.add_argument("--sweep-config", type=str, required=True, help="Path to sweep YAML")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for number of runs (useful for smoke checks)",
    )
    return parser.parse_args()


def _merge_dict(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.sweep_config)
    with sweep_path.open("r", encoding="utf-8") as f:
        sweep = yaml.safe_load(f)

    base_config_path = sweep_path.parent / sweep["base_config"]
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    ablations = sweep.get("ablations", [])
    environments = sweep.get("environments", [])
    seeds = sweep.get("seeds", [])

    if not ablations or not environments or not seeds:
        raise ValueError("Sweep config must include non-empty ablations, environments, and seeds")

    total_planned = len(ablations) * len(environments) * len(seeds)
    print(f"[sweep] planned_runs={total_planned}")

    launched = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ablation in ablations:
        ablation_name = ablation["name"]
        ablation_overrides = ablation.get("overrides", {})

        for env_id in environments:
            for seed in seeds:
                if args.max_runs is not None and launched >= args.max_runs:
                    print(f"[sweep] reached max_runs={args.max_runs}")
                    return

                overrides = {
                    "env": {"env_id": env_id},
                    "train": {"seed_list": [int(seed)]},
                }
                _merge_dict(overrides, ablation_overrides)

                config = load_config(base_config_path, overrides=overrides)
                run_dir = (
                    Path(config.train.output_dir)
                    / "sweep"
                    / timestamp
                    / ablation_name
                    / env_id
                    / f"seed_{seed}"
                )

                print(f"[run] ablation={ablation_name} env={env_id} seed={seed} dir={run_dir}")
                trainer = Trainer(config=config, seed=int(seed), run_dir=run_dir)
                trainer.train()
                launched += 1


if __name__ == "__main__":
    main()
