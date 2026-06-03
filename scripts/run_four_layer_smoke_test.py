#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploration_stack.control.existing_il_sac_controller import ExistingILSACLocalController
from exploration_stack.control.safety_shield import EsdfSafetyShield
from exploration_stack.core import DroneState, LocalCommand, Pose3D, SensorPacket
from exploration_stack.mapping.existing_fallback_backend import MapFromExistingOccupancyBackend
from exploration_stack.planning.frontier_graph_planner import FrontierGraphGlobalExplorer
from exploration_stack.robot.base import RobotAdapter
from exploration_stack.runtime.four_layer_runner import FourLayerRunner
from exploration_stack.semantic.heuristic_reasoner import SemanticHeuristicReasoner
from exploration_stack.slam.sim_ground_truth_backend import SimGroundTruthSlamBackend


class MockRobot(RobotAdapter):
    def __init__(self):
        self.grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        self.state = DroneState(pose=Pose3D(position=(2.5, 2.5, 1.5), frame_id="map"))
        self.last_command = LocalCommand()
        self.step_count = 0

    def reset(self) -> DroneState:
        self.state = DroneState(pose=Pose3D(position=(2.5, 2.5, 1.5), frame_id="map"))
        self.step_count = 0
        return self.state

    def step(self, action: LocalCommand) -> DroneState:
        self.apply_local_command(action)
        x, y, z = self.state.pose.position
        forward = action.velocity_body[0]
        self.step_count += 1
        self.state = DroneState(
            timestamp_s=float(self.step_count),
            pose=Pose3D(position=(x + 0.1 * forward, y, z), frame_id="map", timestamp_s=float(self.step_count)),
            linear_velocity=action.velocity_body,
            angular_velocity=(0.0, 0.0, action.yaw_rate),
        )
        return self.state

    def get_state(self) -> DroneState:
        return self.state

    def get_sensors(self) -> SensorPacket:
        return SensorPacket(
            timestamp_s=float(self.step_count),
            metadata={
                "ground_truth_pose": self.state.pose,
                "occupancy_grid": self.grid,
                "depth_line": [0.75, 0.8, 0.7, 0.9],
                "semantic_line": {"door": 0.2, "corridor": 0.6},
            },
        )

    def apply_local_command(self, cmd: LocalCommand) -> None:
        self.last_command = cmd

    def close(self) -> None:
        return None


def build_mock_runner() -> FourLayerRunner:
    return FourLayerRunner(
        robot=MockRobot(),
        slam_backend=SimGroundTruthSlamBackend(),
        mapping_backend=MapFromExistingOccupancyBackend(map_resolution_m=1.0, frontier_min_cluster_size=2),
        semantic_reasoner=SemanticHeuristicReasoner(),
        global_planner=FrontierGraphGlobalExplorer(),
        local_controller=ExistingILSACLocalController(),
        safety_shield=EsdfSafetyShield(min_clearance_m=0.2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the four-layer architecture smoke test without Isaac/ROS/VLM.")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    runner = build_mock_runner()
    result = runner.run_episode(max_steps=args.steps)
    print(
        "four-layer smoke test passed: "
        f"steps={result.steps}, coverage={result.coverage:.3f}, "
        f"subgoal={result.subgoal.subgoal_id if result.subgoal else None}, reason={result.reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
