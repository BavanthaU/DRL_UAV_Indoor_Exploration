from __future__ import annotations

from dataclasses import dataclass

from exploration_stack.core import DroneState, ExplorationMap, Frontier, SemanticPrior, SlamState, Subgoal
from exploration_stack.planning.astar import astar_path, path_cost
from exploration_stack.planning.base import GlobalExplorer


@dataclass
class FrontierGraphWeights:
    w_info: float = 1.0
    w_sem: float = 1.0
    w_door: float = 0.5
    w_corridor: float = 0.25
    w_cost: float = 0.15
    w_risk: float = 1.0
    w_revisit: float = 0.25
    w_loop: float = 0.25


class FrontierGraphGlobalExplorer(GlobalExplorer):
    """Frontier graph planner with semantic prior and path-cost scoring."""

    def __init__(
        self,
        *,
        weights: FrontierGraphWeights | None = None,
        allow_unknown_paths: bool = False,
        safety_radius_cells: int = 0,
    ):
        self.weights = weights or FrontierGraphWeights()
        self.allow_unknown_paths = allow_unknown_paths
        self.safety_radius_cells = safety_radius_cells

    def select_subgoal(
        self,
        *,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        semantic_prior: SemanticPrior,
        robot_state: DroneState,
    ) -> Subgoal | None:
        candidates = frontiers or exploration_map.frontiers
        if not candidates:
            return None

        grid = exploration_map.occupancy_2d
        start_cell = grid.xy_to_cell((robot_state.pose.position[0], robot_state.pose.position[1]))
        max_gain = max((frontier.information_gain for frontier in candidates), default=1.0) or 1.0

        best: Subgoal | None = None
        for frontier in candidates:
            goal_cell = grid.xy_to_cell(frontier.centroid_xy)
            path = astar_path(
                grid,
                start_cell,
                goal_cell,
                allow_unknown=self.allow_unknown_paths,
                safety_radius_cells=self.safety_radius_cells,
            )
            cost = path_cost(path, resolution_m=grid.resolution_m)
            if path is None:
                continue

            semantic_score = semantic_prior.score_for_frontier(frontier.frontier_id)
            sem = semantic_score.score if semantic_score else 0.0
            risk = semantic_score.risk if semantic_score else 0.0
            door = semantic_score.doorway_likelihood if semantic_score else 0.0
            corridor = semantic_score.corridor_likelihood if semantic_score else 0.0
            revisit_penalty = float(frontier.metadata.get("revisit_penalty", 0.0))
            loop_penalty = float(frontier.metadata.get("loop_penalty", 0.0))
            info = frontier.information_gain / max_gain

            score = (
                self.weights.w_info * info
                + self.weights.w_sem * sem
                + self.weights.w_door * door
                + self.weights.w_corridor * corridor
                - self.weights.w_cost * cost
                - self.weights.w_risk * risk
                - self.weights.w_revisit * revisit_penalty
                - self.weights.w_loop * loop_penalty
            )
            subgoal = Subgoal(
                subgoal_id=frontier.frontier_id,
                frontier_id=frontier.frontier_id,
                position_xy=frontier.centroid_xy,
                score=score,
                reason=(
                    f"info={info:.2f}, sem={sem:.2f}, cost={cost:.2f}, "
                    f"risk={risk:.2f}, door={door:.2f}, corridor={corridor:.2f}"
                ),
                metadata={"path": path, "path_cost_m": cost},
            )
            if best is None or subgoal.score > best.score:
                best = subgoal
        return best
