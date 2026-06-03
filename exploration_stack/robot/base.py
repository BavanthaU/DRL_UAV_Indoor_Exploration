from __future__ import annotations

from abc import ABC, abstractmethod

from exploration_stack.core import DroneState, LocalCommand, SensorPacket


class RobotAdapter(ABC):
    """Common robot interface used by the four-layer runtime."""

    @abstractmethod
    def reset(self) -> DroneState:
        """Reset the robot/backend and return the initial robot state."""

    @abstractmethod
    def step(self, action: LocalCommand) -> DroneState:
        """Advance one backend step using a local command."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Return the current robot state."""

    @abstractmethod
    def get_sensors(self) -> SensorPacket:
        """Return the latest synchronized sensor packet."""

    @abstractmethod
    def apply_local_command(self, cmd: LocalCommand) -> None:
        """Apply a low-level/local command without advancing the backend."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
