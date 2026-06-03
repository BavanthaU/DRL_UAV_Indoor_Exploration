from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from exploration_stack.core import (
    DroneState,
    ExplorationMap,
    Frontier,
    SemanticPrior,
    SensorPacket,
    SlamState,
    VlmObservationPacket,
    dataclass_to_dict,
)
from exploration_stack.semantic.base import SemanticReasoner
from exploration_stack.semantic.schema import semantic_prior_from_vlm_output, validate_vlm_output

Provider = Callable[[VlmObservationPacket], dict[str, Any] | str]


class VlmSemanticReasoner(SemanticReasoner):
    """Optional low-rate/offline VLM frontier scorer with JSON validation."""

    def __init__(
        self,
        *,
        provider: Provider | None = None,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.provider = provider
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def infer(
        self,
        *,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        robot_state: DroneState,
    ) -> SemanticPrior:
        observation = VlmObservationPacket(
            sensor_packet=sensor_packet,
            slam_state=slam_state,
            exploration_map=exploration_map,
            frontiers=frontiers,
            robot_state=robot_state,
        )
        cache_key = self._cache_key(observation)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return semantic_prior_from_vlm_output(validate_vlm_output(cached))

        raw_output = self._call_provider(observation)
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        validated = validate_vlm_output(data)
        self._write_cache(cache_key, data)
        return semantic_prior_from_vlm_output(validated)

    def _call_provider(self, observation: VlmObservationPacket) -> dict[str, Any] | str:
        if self.provider is not None:
            return self.provider(observation)
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "VlmSemanticReasoner needs a provider callable for tests/offline "
                "labels, or optional VLM dependencies such as transformers for "
                "Qwen2.5-VL/SmolVLM. Install the VLM extras and keep this "
                "reasoner low-rate/offline."
            ) from exc
        raise NotImplementedError(
            "Direct VLM inference is not wired yet. Provide a callable that "
            "returns the strict JSON schema, or implement model loading for "
            f"{self.model_name or 'the configured VLM'}."
        )

    def _cache_key(self, observation: VlmObservationPacket) -> str:
        payload = json.dumps(dataclass_to_dict(observation), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
