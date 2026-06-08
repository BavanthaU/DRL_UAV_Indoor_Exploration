from __future__ import annotations

from dataclasses import dataclass
from math import radians

from exploration_stack.cpp_accel import grid_planning
from exploration_stack.depth_hierarchical.map_memory import OCCUPIED, UNKNOWN


@dataclass
class ActiveMappingDuringTransitConfig:
    yaw_candidates_deg: tuple[int, ...] = (-60, -30, 0, 30, 60)
    raycast_max_cells: int = 24
    obstacle_risk_threshold: float = 0.5


class ActiveMappingDuringTransit:
    def __init__(self, cfg: ActiveMappingDuringTransitConfig | None = None):
        self.cfg = cfg or ActiveMappingDuringTransitConfig()

    def choose_yaw(self, known_grid, agent_cell, current_yaw_rad: float, obstacle_risk: float = 0.0):
        best_yaw = current_yaw_rad
        best_gain = -1.0
        for yaw_deg in self.cfg.yaw_candidates_deg:
            yaw = current_yaw_rad + radians(float(yaw_deg))
            gain = grid_planning.raycast_unknown_gain(
                known_grid,
                agent_cell,
                yaw,
                max_range_cells=self.cfg.raycast_max_cells,
                unknown_value=UNKNOWN,
                occupied_value=OCCUPIED,
            )
            if obstacle_risk < self.cfg.obstacle_risk_threshold and yaw_deg == 0:
                gain += 0.25
            if gain > best_gain:
                best_yaw = yaw
                best_gain = float(gain)
        return best_yaw, best_gain
