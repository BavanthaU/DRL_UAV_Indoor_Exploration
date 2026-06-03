from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class StructuredRunLogger:
    """JSONL/CSV logger for PPO exploration runs."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_fields: list[str] | None = None

    def write_config(self, config: dict[str, Any], *, git_commit: str | None = None) -> None:
        payload = {"config": config, "git_commit": git_commit}
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True, default=float) + "\n")
        flat = {key: self._scalar(value) for key, value in row.items() if self._is_scalar(value)}
        if self._csv_fields is None:
            self._csv_fields = list(flat.keys())
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self._csv_fields)
                writer.writeheader()
                writer.writerow(flat)
        else:
            with self.csv_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self._csv_fields, extrasaction="ignore")
                writer.writerow(flat)

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool, str))

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        return value

