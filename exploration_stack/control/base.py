from __future__ import annotations

from abc import ABC, abstractmethod

from exploration_stack.core import DroneState, ExplorationMap, LocalCommand, SensorPacket, SlamState, Subgoal


class LocalController(ABC):
    """Policy-rate local controller."""

    @abstractmethod
    def act(
        self,
        *,
        robot_state: DroneState,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        subgoal: Subgoal | None,
    ) -> LocalCommand:
        """Produce a local command for the active subgoal."""


class SafetyShield(ABC):
    """Policy-rate command filter."""

    @abstractmethod
    def filter(
        self,
        command: LocalCommand,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
    ) -> LocalCommand:
        """Return a safe command."""
