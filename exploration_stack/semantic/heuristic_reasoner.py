from __future__ import annotations

import math
from typing import Any

from exploration_stack.core import DroneState, ExplorationMap, Frontier, FrontierScore, SemanticPrior, SensorPacket, SlamState
from exploration_stack.semantic.base import SemanticReasoner


class SemanticHeuristicReasoner(SemanticReasoner):
    """Dependency-free frontier scorer using map, depth, and semantic-line cues."""

    def __init__(
        self,
        *,
        w_info: float = 0.55,
        w_depth_open: float = 0.15,
        w_door: float = 0.15,
        w_corridor: float = 0.10,
        w_risk: float = 0.20,
        max_expected_gain: float = 50.0,
    ):
        self.w_info = w_info
        self.w_depth_open = w_depth_open
        self.w_door = w_door
        self.w_corridor = w_corridor
        self.w_risk = w_risk
        self.max_expected_gain = max_expected_gain

    def infer(
        self,
        *,
        sensor_packet: SensorPacket,
        slam_state: SlamState,
        exploration_map: ExplorationMap,
        frontiers: list[Frontier],
        robot_state: DroneState,
    ) -> SemanticPrior:
        candidates = frontiers or exploration_map.frontiers
        depth_open = self._depth_open_score(sensor_packet.metadata.get("depth_line"))
        door_likelihood = self._semantic_keyword_score(sensor_packet.metadata.get("semantic_line"), "door")
        corridor_likelihood = max(
            self._semantic_keyword_score(sensor_packet.metadata.get("semantic_line"), "corridor"),
            self._corridor_from_frontier_shape(candidates),
        )

        scores: list[FrontierScore] = []
        for frontier in candidates:
            risk = self._risk_for_frontier(exploration_map, frontier)
            info_score = min(1.0, frontier.information_gain / max(self.max_expected_gain, 1.0))
            score = (
                self.w_info * info_score
                + self.w_depth_open * depth_open
                + self.w_door * door_likelihood
                + self.w_corridor * corridor_likelihood
                - self.w_risk * risk
            )
            score = max(0.0, min(1.0, score))
            reason = (
                f"info={info_score:.2f}, depth_open={depth_open:.2f}, "
                f"door={door_likelihood:.2f}, corridor={corridor_likelihood:.2f}, risk={risk:.2f}"
            )
            scores.append(
                FrontierScore(
                    frontier_id=frontier.frontier_id,
                    score=score,
                    reason=reason,
                    risk=risk,
                    expected_information_gain=frontier.information_gain,
                    doorway_likelihood=door_likelihood,
                    corridor_likelihood=corridor_likelihood,
                )
            )

        scores.sort(key=lambda item: item.score, reverse=True)
        recommended = scores[0].frontier_id if scores else None
        uncertainty = 0.35 if sensor_packet.metadata.get("semantic_line") is not None else 0.55
        return SemanticPrior(
            scene_summary="Heuristic map/depth/semantic frontier prior.",
            room_type_guess=sensor_packet.metadata.get("room_type_guess", "unknown"),
            frontier_scores=scores,
            recommended_subgoal_id=recommended,
            uncertainty=uncertainty,
            planner_prior={"source": "semantic_heuristic"},
        )

    @staticmethod
    def _depth_open_score(depth_line: Any) -> float:
        if depth_line is None:
            return 0.0
        values = [float(value) for value in _flatten(depth_line)]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            return 0.0
        return max(0.0, min(1.0, sum(values) / len(values)))

    @staticmethod
    def _semantic_keyword_score(semantic_line: Any, keyword: str) -> float:
        if semantic_line is None:
            return 0.0
        if isinstance(semantic_line, str):
            return 1.0 if keyword.lower() in semantic_line.lower() else 0.0
        if isinstance(semantic_line, dict):
            return float(semantic_line.get(keyword, 0.0))
        labels = [str(item).lower() for item in _flatten(semantic_line)]
        if not labels:
            return 0.0
        return labels.count(keyword.lower()) / len(labels)

    @staticmethod
    def _corridor_from_frontier_shape(frontiers: list[Frontier]) -> float:
        if not frontiers:
            return 0.0
        elongated = 0
        for frontier in frontiers:
            if len(frontier.cells) < 3:
                continue
            rows = [cell[0] for cell in frontier.cells]
            cols = [cell[1] for cell in frontier.cells]
            row_span = max(rows) - min(rows) + 1
            col_span = max(cols) - min(cols) + 1
            ratio = max(row_span, col_span) / max(1, min(row_span, col_span))
            elongated += ratio >= 3.0
        return elongated / max(1, len(frontiers))

    @staticmethod
    def _risk_for_frontier(exploration_map: ExplorationMap, frontier: Frontier) -> float:
        if exploration_map.esdf is None or not exploration_map.esdf.distances_m:
            return 0.0
        esdf = exploration_map.esdf.distances_m
        risks = []
        for row, col in frontier.cells:
            if 0 <= row < len(esdf) and 0 <= col < len(esdf[0]):
                distance = float(esdf[row][col])
                if math.isfinite(distance):
                    risks.append(max(0.0, 1.0 - min(distance, 1.0)))
        return sum(risks) / len(risks) if risks else 0.0


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_flatten(item))
        return output
    return [value]
