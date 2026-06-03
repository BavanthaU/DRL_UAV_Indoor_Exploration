from __future__ import annotations

from abc import ABC, abstractmethod

from exploration_stack.core import DroneState, ExplorationMap, Frontier, SemanticPrior, SlamState, Subgoal


class GlobalExplorer(ABC):
    """Low-rate global subgoal selector."""

    @abstractmethod
    def select_subgoal(
        self,
        *,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        semantic_prior: SemanticPrior,
        robot_state: DroneState,
    ) -> Subgoal | None:
        """Select the next global subgoal."""
