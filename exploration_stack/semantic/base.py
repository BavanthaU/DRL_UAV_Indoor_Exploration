from __future__ import annotations

from abc import ABC, abstractmethod

from exploration_stack.core import DroneState, ExplorationMap, Frontier, SemanticPrior, SensorPacket, SlamState


class SemanticReasoner(ABC):
    """Low-rate semantic prior provider for frontier scoring."""

    @abstractmethod
    def infer(
        self,
        *,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        robot_state: DroneState,
    ) -> SemanticPrior:
        """Infer semantic/frontier priors for the global planner."""
