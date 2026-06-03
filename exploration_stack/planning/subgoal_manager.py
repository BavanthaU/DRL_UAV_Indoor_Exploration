from __future__ import annotations

import math
from dataclasses import dataclass, field

from exploration_stack.core import DroneState, Subgoal


@dataclass
class SubgoalManager:
    reached_radius_m: float = 0.5
    max_no_coverage_steps: int = 20
    max_failures_per_frontier: int = 3
    active_subgoal: Subgoal | None = None
    no_coverage_steps: int = 0
    failed_frontiers: dict[int, int] = field(default_factory=dict)

    def should_replan(
        self,
        *,
        robot_state: DroneState,
        coverage_delta: float,
        path_blocked: bool = False,
        loop_detected: bool = False,
    ) -> bool:
        if self.active_subgoal is None:
            return True
        if self._is_reached(robot_state, self.active_subgoal):
            return True
        if path_blocked or loop_detected:
            self.mark_failed(self.active_subgoal)
            return True
        if coverage_delta <= 0.0:
            self.no_coverage_steps += 1
        else:
            self.no_coverage_steps = 0
        return self.no_coverage_steps >= self.max_no_coverage_steps

    def set_subgoal(self, subgoal: Subgoal | None) -> None:
        self.active_subgoal = subgoal
        self.no_coverage_steps = 0

    def mark_failed(self, subgoal: Subgoal) -> None:
        if subgoal.frontier_id is None:
            return
        self.failed_frontiers[subgoal.frontier_id] = self.failed_frontiers.get(subgoal.frontier_id, 0) + 1

    def is_stale(self, subgoal: Subgoal) -> bool:
        if subgoal.frontier_id is None:
            return False
        return self.failed_frontiers.get(subgoal.frontier_id, 0) >= self.max_failures_per_frontier

    def _is_reached(self, robot_state: DroneState, subgoal: Subgoal) -> bool:
        dx = robot_state.pose.position[0] - subgoal.position_xy[0]
        dy = robot_state.pose.position[1] - subgoal.position_xy[1]
        return math.hypot(dx, dy) <= self.reached_radius_m
