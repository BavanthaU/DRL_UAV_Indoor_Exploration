from __future__ import annotations

from exploration_stack.core import EsdfMap, ExplorationMap, OccupancyGrid2D, Pose3D, SemanticGrid, SensorPacket
from exploration_stack.mapping.base import MappingBackend


class RtabmapMappingBackend(MappingBackend):
    """Optional ROS 2 topic-wrapper shell for RTAB-Map style mapping."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "RtabmapMappingBackend requires ROS 2 Python bindings and "
                "RTAB-Map ROS. Install/activate ROS 2, or use "
                "mapping_backend=existing_fallback for smoke tests."
            ) from exc
        raise NotImplementedError(
            "RtabmapMappingBackend is an optional wrapper skeleton. Configure "
            "map, odometry, and semantic topics before selecting it."
        )

    def update(self, sensor_packet: SensorPacket, pose: Pose3D) -> ExplorationMap:
        raise NotImplementedError

    def get_esdf(self) -> EsdfMap | None:
        raise NotImplementedError

    def get_occupancy_2d(self) -> OccupancyGrid2D:
        raise NotImplementedError

    def get_semantic_grid(self) -> SemanticGrid | None:
        raise NotImplementedError
