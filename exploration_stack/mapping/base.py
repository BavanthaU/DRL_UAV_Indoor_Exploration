from __future__ import annotations

from abc import ABC, abstractmethod

from exploration_stack.core import EsdfMap, ExplorationMap, OccupancyGrid2D, Pose3D, SemanticGrid, SensorPacket


class MappingBackend(ABC):
    """Common interface for Nvblox, RTAB-Map, and existing occupancy fallback."""

    @abstractmethod
    def update(self, sensor_packet: SensorPacket, pose: Pose3D) -> ExplorationMap:
        """Update the map from a sensor packet and pose estimate."""

    @abstractmethod
    def get_esdf(self) -> EsdfMap | None:
        """Return the latest ESDF map if available."""

    @abstractmethod
    def get_occupancy_2d(self) -> OccupancyGrid2D:
        """Return the latest 2D exploration occupancy grid."""

    @abstractmethod
    def get_semantic_grid(self) -> SemanticGrid | None:
        """Return the latest semantic grid if available."""
