from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from exploration_stack.core import ExplorationMap, Pose3D, SensorPacket, SlamState


class SlamBackend(ABC):
    """Common interface for visual SLAM, ground-truth debug, and fallbacks."""

    @abstractmethod
    def reset(self) -> None:
        """Reset SLAM state."""

    @abstractmethod
    def update(self, sensor_packet: SensorPacket) -> SlamState:
        """Update SLAM from synchronized sensors and return tracking state."""

    @abstractmethod
    def get_pose(self) -> Pose3D:
        """Return the latest pose estimate."""

    @abstractmethod
    def get_map(self) -> ExplorationMap | None:
        """Return a backend-owned map if available."""

    @abstractmethod
    def save_map(self, path: str | Path) -> None:
        """Persist map/state to disk."""

    @abstractmethod
    def load_map(self, path: str | Path) -> None:
        """Load map/state from disk."""
