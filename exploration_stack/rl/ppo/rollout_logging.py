from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


class StructuredRunLogger:
    """JSONL/CSV logger for PPO exploration runs with optional W&B upload."""

    def __init__(self, run_dir: str | Path, *, wandb_config: dict[str, Any] | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self._csv_fields: list[str] | None = None
        self._wandb_config = wandb_config or {}
        self._wandb_run = None
        self._wandb = None

    def write_config(self, config: dict[str, Any], *, git_commit: str | None = None) -> None:
        payload = {"config": config, "git_commit": git_commit}
        config_path = self.run_dir / "config.json"
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._init_wandb(payload)
        self.log_artifact(config_path, artifact_type="config", enabled_key="log_config", aliases=["latest"])

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
        if self._wandb_run is not None:
            self._wandb_run.log(flat, step=int(step))

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        name: str | None = None,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        enabled_key: str | None = None,
    ) -> None:
        if not self.should_log_artifact(enabled_key):
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            return
        artifact_name = self._artifact_name(name or f"{self.run_dir.name}-{artifact_type}-{artifact_path.stem}")
        artifact = self._wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        self._wandb_run.log_artifact(artifact, aliases=aliases)

    def log_image(
        self,
        key: str,
        path: str | Path,
        *,
        step: int = 0,
        caption: str | None = None,
        enabled_key: str | None = None,
    ) -> None:
        if not self.should_log_artifact(enabled_key):
            return
        image_path = Path(path)
        if not image_path.exists():
            return
        self._wandb_run.log({key: self._wandb.Image(str(image_path), caption=caption)}, step=int(step))

    def wandb_config_value(self, key: str, default: Any = None) -> Any:
        return self._wandb_config.get(key, default)

    def should_log_artifact(self, enabled_key: str | None = None) -> bool:
        if self._wandb_run is None:
            return False
        if not bool(self._wandb_config.get("log_artifacts", True)):
            return False
        if enabled_key is None:
            return True
        return bool(self._wandb_config.get(enabled_key, True))

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _init_wandb(self, payload: dict[str, Any]) -> None:
        if self._wandb_run is not None:
            return
        if not bool(self._wandb_config.get("enabled", False)):
            return
        mode = self._wandb_config.get("mode", "online")
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but the `wandb` package is not installed. "
                "Install wandb or set wandb.enabled=false / --wandb_mode disabled."
            ) from exc
        project = self._wandb_config.get("project") or "vlm-ppo-uav-exploration"
        name = self._wandb_config.get("name") or self.run_dir.name
        self._wandb = wandb
        self._wandb_run = wandb.init(
            project=project,
            entity=self._wandb_config.get("entity"),
            name=name,
            group=self._wandb_config.get("group"),
            tags=self._wandb_config.get("tags"),
            mode=mode,
            dir=str(self.run_dir),
            config=payload,
            resume=self._wandb_config.get("resume", "allow"),
        )

    @staticmethod
    def _artifact_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")[:128] or "artifact"

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool, str))

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        return value
