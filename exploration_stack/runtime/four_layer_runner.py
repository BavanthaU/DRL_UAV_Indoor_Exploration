from __future__ import annotations

from dataclasses import dataclass, field

from exploration_stack.control.base import LocalController, SafetyShield
from exploration_stack.core import SemanticPrior, Subgoal
from exploration_stack.mapping.base import MappingBackend
from exploration_stack.mapping.frontier_utils import coverage_fraction, frontiers_from_occupancy_grid
from exploration_stack.planning.base import GlobalExplorer
from exploration_stack.planning.subgoal_manager import SubgoalManager
from exploration_stack.robot.base import RobotAdapter
from exploration_stack.semantic.base import SemanticReasoner
from exploration_stack.slam.base import SlamBackend


@dataclass
class EpisodeResult:
    steps: int
    coverage: float
    subgoal: Subgoal | None
    records: list[dict] = field(default_factory=list)
    reason: str = "max_steps"


class FourLayerRunner:
    """Orchestrates robot, SLAM/map, semantic prior, planner, local control."""

    def __init__(
        self,
        *,
        robot: RobotAdapter,
        slam_backend: SlamBackend,
        mapping_backend: MappingBackend,
        semantic_reasoner: SemanticReasoner,
        global_planner: GlobalExplorer,
        local_controller: LocalController,
        safety_shield: SafetyShield,
        subgoal_manager: SubgoalManager | None = None,
        advance_robot_with_step: bool = True,
    ):
        self.robot = robot
        self.slam_backend = slam_backend
        self.mapping_backend = mapping_backend
        self.semantic_reasoner = semantic_reasoner
        self.global_planner = global_planner
        self.local_controller = local_controller
        self.safety_shield = safety_shield
        self.subgoal_manager = subgoal_manager or SubgoalManager()
        self.advance_robot_with_step = advance_robot_with_step

    def run_episode(self, *, max_steps: int = 100) -> EpisodeResult:
        robot_state = self.robot.reset()
        self.slam_backend.reset()
        active_prior = SemanticPrior()
        previous_coverage = 0.0
        records: list[dict] = []
        reason = "max_steps"

        for step in range(max_steps):
            sensor_packet = self.robot.get_sensors()
            robot_state = self.robot.get_state()
            slam_state = self.slam_backend.update(sensor_packet)
            exploration_map = self.mapping_backend.update(sensor_packet, slam_state.pose)
            frontiers = exploration_map.frontiers or frontiers_from_occupancy_grid(exploration_map.occupancy_2d)
            coverage = coverage_fraction(exploration_map.occupancy_2d)
            coverage_delta = coverage - previous_coverage

            if self.subgoal_manager.should_replan(
                robot_state=robot_state,
                coverage_delta=coverage_delta,
            ):
                active_prior = self.semantic_reasoner.infer(
                    sensor_packet=sensor_packet,
                    slam_state=slam_state,
                    exploration_map=exploration_map,
                    frontiers=frontiers,
                    robot_state=robot_state,
                )
                subgoal = self.global_planner.select_subgoal(
                    slam_state=slam_state,
                    exploration_map=exploration_map,
                    frontiers=frontiers,
                    semantic_prior=active_prior,
                    robot_state=robot_state,
                )
                self.subgoal_manager.set_subgoal(subgoal)
                if subgoal is None:
                    reason = "no_subgoal"
                    records.append({"step": step, "coverage": coverage, "reason": reason})
                    break

            local_cmd = self.local_controller.act(
                robot_state=robot_state,
                sensor_packet=sensor_packet,
                slam_state=slam_state,
                subgoal=self.subgoal_manager.active_subgoal,
            )
            safe_cmd = self.safety_shield.filter(local_cmd, slam_state, exploration_map)
            if self.advance_robot_with_step:
                robot_state = self.robot.step(safe_cmd)
            else:
                self.robot.apply_local_command(safe_cmd)

            records.append(
                {
                    "step": step,
                    "coverage": coverage,
                    "coverage_delta": coverage_delta,
                    "num_frontiers": len(frontiers),
                    "subgoal_id": self.subgoal_manager.active_subgoal.subgoal_id
                    if self.subgoal_manager.active_subgoal
                    else None,
                    "semantic_recommended_subgoal_id": active_prior.recommended_subgoal_id,
                    "safety_intervention": safe_cmd.metadata.get("safety_intervention"),
                }
            )
            previous_coverage = coverage

        return EpisodeResult(
            steps=len(records),
            coverage=previous_coverage,
            subgoal=self.subgoal_manager.active_subgoal,
            records=records,
            reason=reason,
        )
