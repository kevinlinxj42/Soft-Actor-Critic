#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _auc(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float(ys[-1] if len(ys) else 0.0)
    return float(np.trapz(ys, xs) / max(xs[-1] - xs[0], 1.0))


def summarize_run(eval_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(eval_path)
    if not rows:
        return {}

    steps = np.array([r["step"] for r in rows], dtype=np.float64)
    returns = np.array([r["eval_return_mean"] for r in rows], dtype=np.float64)

    return {
        "env_id": eval_path.parent.parent.name,
        "run": eval_path.parent.name,
        "final_return": float(returns[-1]),
        "best_return": float(np.max(returns)),
        "auc_return": _auc(steps, returns),
        "num_eval_points": int(len(rows)),
    }


def mean_ci95(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0
    stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, 1.96 * stderr


def format_table(grouped: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| env_id | n | final_return (mean ± 95% CI) | best_return (mean ± 95% CI) | auc_return (mean ± 95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]

    for env_id in sorted(grouped):
        entries = grouped[env_id]
        final_mean, final_ci = mean_ci95([e["final_return"] for e in entries])
        best_mean, best_ci = mean_ci95([e["best_return"] for e in entries])
        auc_mean, auc_ci = mean_ci95([e["auc_return"] for e in entries])
        lines.append(
            f"| {env_id} | {len(entries)} | {final_mean:.2f} ± {final_ci:.2f} | "
            f"{best_mean:.2f} ± {best_ci:.2f} | {auc_mean:.2f} ± {auc_ci:.2f} |"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SAC run metrics across seeds")
    parser.add_argument(
        "--runs-root",
        type=str,
        required=True,
        help="Root directory containing run subdirectories with eval_metrics.jsonl files",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional summary JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)

    summaries: list[dict[str, Any]] = []
    for eval_path in runs_root.rglob("eval_metrics.jsonl"):
        summary = summarize_run(eval_path)
        if summary:
            summaries.append(summary)

    if not summaries:
        raise SystemExit(f"No eval_metrics.jsonl files found under {runs_root}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        grouped[row["env_id"]].append(row)

    table = format_table(grouped)
    print(table)

    payload = {
        "runs_root": str(runs_root),
        "table_markdown": table,
        "per_run": summaries,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
