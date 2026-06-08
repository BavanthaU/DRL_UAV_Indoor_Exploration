from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from exploration_stack.cpp_accel import grid_planning
from exploration_stack.depth_hierarchical.map_memory import Candidate, OCCUPIED


@dataclass
class AStarTransitNavigatorConfig:
    obstacle_inflation_cells: int = 2
    allow_diagonal: bool = False
    max_replans: int = 3


@dataclass
class TransitPlan:
    selected_candidate_id: int
    path: list[tuple[int, int]]
    cost: float
    reachable: bool
    replans: int = 0
    aborted: bool = False


class AStarTransitNavigator:
    """A* transit planner that never chooses the candidate itself."""

    def __init__(self, cfg: AStarTransitNavigatorConfig | None = None):
        self.cfg = cfg or AStarTransitNavigatorConfig()

    def plan_to_selected_candidate(self, start_cell: tuple[int, int], selected_candidate: Candidate, known_grid) -> TransitPlan:
        occupancy = self._inflate_obstacles(known_grid)
        path, cost = grid_planning.astar_grid(
            start_cell,
            (selected_candidate.row, selected_candidate.col),
            occupancy,
            allow_diagonal=self.cfg.allow_diagonal,
        )
        return TransitPlan(
            selected_candidate_id=selected_candidate.candidate_id,
            path=list(path),
            cost=float(cost),
            reachable=bool(path),
        )

    def replan_if_blocked(self, start_cell, selected_candidate: Candidate, known_grid, previous: TransitPlan) -> TransitPlan:
        if previous.replans >= self.cfg.max_replans:
            previous.aborted = True
            return previous
        new_plan = self.plan_to_selected_candidate(start_cell, selected_candidate, known_grid)
        new_plan.replans = previous.replans + 1
        if not new_plan.reachable and new_plan.replans >= self.cfg.max_replans:
            new_plan.aborted = True
        return new_plan

    def _inflate_obstacles(self, known_grid):
        if np is None:
            raise RuntimeError("AStarTransitNavigator requires numpy.")
        grid = np.asarray(known_grid)
        occupancy = np.zeros_like(grid, dtype=np.int16)
        occupied = np.argwhere(grid == OCCUPIED)
        for row, col in occupied:
            r0 = max(0, int(row) - self.cfg.obstacle_inflation_cells)
            r1 = min(grid.shape[0], int(row) + self.cfg.obstacle_inflation_cells + 1)
            c0 = max(0, int(col) - self.cfg.obstacle_inflation_cells)
            c1 = min(grid.shape[1], int(col) + self.cfg.obstacle_inflation_cells + 1)
            occupancy[r0:r1, c0:c1] = 2
        return occupancy.tolist()
