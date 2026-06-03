from __future__ import annotations

import logging
import math

from exploration_stack.control.base import SafetyShield
from exploration_stack.core import ExplorationMap, LocalCommand, SlamState

LOGGER = logging.getLogger(__name__)


class EsdfSafetyShield(SafetyShield):
    """Simple ESDF/occupancy command filter for local safety."""

    def __init__(
        self,
        *,
        min_clearance_m: float = 0.45,
        slow_forward_scale: float = 0.25,
        avoidance_yaw_rate: float = 0.4,
    ):
        self.min_clearance_m = min_clearance_m
        self.slow_forward_scale = slow_forward_scale
        self.avoidance_yaw_rate = avoidance_yaw_rate
        self.interventions: list[dict] = []

    def filter(
        self,
        command: LocalCommand,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
    ) -> LocalCommand:
        if command.stop:
            return command
        clearance = self._clearance_at_pose(slam_state, exploration_map)
        if clearance is None or clearance >= self.min_clearance_m:
            return command

        forward, lateral, vertical = command.velocity_body
        if clearance <= 0.0:
            safe_command = LocalCommand(
                velocity_body=(0.0, 0.0, 0.0),
                yaw_rate=self.avoidance_yaw_rate,
                metadata={**command.metadata, "safety_intervention": "rotate_away_from_obstacle"},
            )
        else:
            safe_command = LocalCommand(
                velocity_body=(min(forward, forward * self.slow_forward_scale), lateral, vertical),
                yaw_rate=command.yaw_rate,
                metadata={**command.metadata, "safety_intervention": "reduced_forward_velocity"},
            )
        event = {
            "clearance_m": clearance,
            "threshold_m": self.min_clearance_m,
            "intervention": safe_command.metadata["safety_intervention"],
        }
        self.interventions.append(event)
        LOGGER.info("Safety shield intervention: %s", event)
        return safe_command

    def _clearance_at_pose(self, slam_state: SlamState, exploration_map: ExplorationMap) -> float | None:
        if exploration_map.esdf is not None and exploration_map.esdf.distances_m:
            esdf = exploration_map.esdf.distances_m
            grid = exploration_map.occupancy_2d
            row, col = grid.xy_to_cell((slam_state.pose.position[0], slam_state.pose.position[1]))
            if 0 <= row < len(esdf) and 0 <= col < len(esdf[0]):
                value = float(esdf[row][col])
                return value if math.isfinite(value) else None
        collision = exploration_map.metadata.get("closest_distance_m")
        return float(collision) if collision is not None else None
