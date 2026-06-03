from __future__ import annotations

import logging
from typing import Any

from exploration_stack.core import EsdfMap, ExplorationMap, OccupancyGrid2D, Pose3D, SemanticGrid, SensorPacket
from exploration_stack.mapping.base import MappingBackend
from exploration_stack.mapping.frontier_utils import frontiers_from_occupancy_grid, normalize_occupancy_grid

LOGGER = logging.getLogger(__name__)


class MapFromExistingOccupancyBackend(MappingBackend):
    """Fallback mapper using the current repo occupancy grid.

    This is not a final SLAM/mapping backend. It exists so the four-layer
    runtime can be smoke-tested before Isaac ROS Visual SLAM and Nvblox are
    wired in.
    """

    def __init__(
        self,
        *,
        map_resolution_m: float = 0.10,
        frontier_min_cluster_size: int = 5,
        env_index: int = 0,
    ):
        self.map_resolution_m = map_resolution_m
        self.frontier_min_cluster_size = frontier_min_cluster_size
        self.env_index = env_index
        self._map = ExplorationMap(
            occupancy_2d=OccupancyGrid2D(resolution_m=map_resolution_m),
            metadata={"source": "existing_fallback"},
        )
        LOGGER.warning(
            "MapFromExistingOccupancyBackend selected. This uses the current "
            "repo occupancy approximation and should be treated as fallback."
        )

    def update(self, sensor_packet: SensorPacket, pose: Pose3D) -> ExplorationMap:
        raw_grid = sensor_packet.metadata.get("occupancy_grid")
        if raw_grid is None:
            return self._map

        grid = self._to_grid(raw_grid)
        if grid and isinstance(grid[0], list) and grid[0] and isinstance(grid[0][0], list):
            grid = grid[min(self.env_index, len(grid) - 1)]
        normalized = normalize_occupancy_grid(grid, source="existing")
        occupancy = OccupancyGrid2D(
            cells=normalized,
            resolution_m=self.map_resolution_m,
            origin_xy=sensor_packet.metadata.get("map_origin_xy", (0.0, 0.0)),
            frame_id=pose.frame_id,
        )
        frontiers = frontiers_from_occupancy_grid(
            occupancy,
            min_cluster_size=self.frontier_min_cluster_size,
        )
        self._map = ExplorationMap(
            occupancy_2d=occupancy,
            esdf=self._map.esdf,
            semantic_grid=self._map.semantic_grid,
            frontiers=frontiers,
            room_graph=self._map.room_graph,
            metadata={"source": "existing_fallback", "pose_frame_id": pose.frame_id},
        )
        return self._map

    def get_esdf(self) -> EsdfMap | None:
        return self._map.esdf

    def get_occupancy_2d(self) -> OccupancyGrid2D:
        return self._map.occupancy_2d

    def get_semantic_grid(self) -> SemanticGrid | None:
        return self._map.semantic_grid

    @staticmethod
    def _to_grid(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
