from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class MetricLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.train_jsonl = self.run_dir / "train_metrics.jsonl"
        self.eval_jsonl = self.run_dir / "eval_metrics.jsonl"
        self.train_csv = self.run_dir / "train_metrics.csv"
        self.eval_csv = self.run_dir / "eval_metrics.csv"

        self._train_fields: list[str] | None = None
        self._eval_fields: list[str] | None = None

    def _write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_csv(self, path: Path, payload: dict[str, Any], fields_cache_name: str) -> None:
        fields = getattr(self, fields_cache_name)
        if fields is None:
            fields = sorted(payload.keys())
            setattr(self, fields_cache_name, fields)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(payload)
        else:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(payload)

    def log_train(self, step: int, metrics: dict[str, float]) -> None:
        payload: dict[str, Any] = {"step": step, "wall_time": time.time(), **metrics}
        self._write_jsonl(self.train_jsonl, payload)
        self._write_csv(self.train_csv, payload, "_train_fields")

    def log_eval(self, step: int, metrics: dict[str, float]) -> None:
        payload: dict[str, Any] = {"step": step, "wall_time": time.time(), **metrics}
        self._write_jsonl(self.eval_jsonl, payload)
        self._write_csv(self.eval_csv, payload, "_eval_fields")

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
