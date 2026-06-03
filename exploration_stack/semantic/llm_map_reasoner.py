from __future__ import annotations

import json
from typing import Any, Callable

from exploration_stack.core import DroneState, ExplorationMap, Frontier, SemanticPrior, SensorPacket, SlamState
from exploration_stack.semantic.base import SemanticReasoner
from exploration_stack.semantic.schema import semantic_prior_from_vlm_output, validate_vlm_output

Provider = Callable[[dict[str, Any]], dict[str, Any] | str]


class LlmMapReasoner(SemanticReasoner):
    """Structured-map-only LLM reasoner shell.

    This is useful for testing semantic priors without sending camera images to
    a model. A provider callable can be a local model, a mock, or a future
    offline labeling service.
    """

    def __init__(self, *, provider: Provider | None = None):
        self.provider = provider

    def infer(
        self,
        *,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        robot_state: DroneState,
    ) -> SemanticPrior:
        if self.provider is None:
            raise RuntimeError(
                "LlmMapReasoner requires a provider callable that returns the "
                "strict frontier-scoring JSON schema. No cloud API is required "
                "or assumed by this repository."
            )
        payload = {
            "pose": slam_state.pose,
            "coverage": exploration_map.metadata.get("coverage"),
            "frontiers": frontiers,
            "robot_state": robot_state,
        }
        raw_output = self.provider(payload)
        data = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        return semantic_prior_from_vlm_output(validate_vlm_output(data))
