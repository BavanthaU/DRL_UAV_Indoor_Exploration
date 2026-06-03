from __future__ import annotations

import tempfile
import unittest

from exploration_stack.core import (
    DroneState,
    ExplorationMap,
    Frontier,
    OccupancyGrid2D,
    Pose3D,
    SensorPacket,
    SlamState,
)
from exploration_stack.semantic.heuristic_reasoner import SemanticHeuristicReasoner
from exploration_stack.semantic.schema import validate_vlm_output
from exploration_stack.semantic.vlm_reasoner import VlmSemanticReasoner


def valid_vlm_json() -> dict:
    return {
        "scene_summary": "office corridor",
        "room_type_guess": "corridor",
        "visible_structures": {"doors": [], "corridors": [], "open_space": [], "blocked_regions": []},
        "frontier_scores": [
            {
                "frontier_id": 1,
                "score": 0.8,
                "reason": "large open frontier",
                "risk": 0.1,
                "expected_information_gain": 12.0,
                "doorway_likelihood": 0.2,
                "corridor_likelihood": 0.7,
            }
        ],
        "recommended_subgoal_id": 1,
        "uncertainty": 0.2,
    }


class SemanticReasonerTest(unittest.TestCase):
    def setUp(self):
        self.frontier = Frontier(frontier_id=1, cells=[(1, 1), (1, 2), (1, 3)], centroid_xy=(2.0, 1.0), information_gain=12.0)
        self.map = ExplorationMap(
            occupancy_2d=OccupancyGrid2D(cells=[[0, 0, 0], [0, 1, 0], [0, 0, 0]], resolution_m=1.0),
            frontiers=[self.frontier],
        )
        self.sensor = SensorPacket(metadata={"depth_line": [0.8, 0.9], "semantic_line": {"door": 0.2, "corridor": 0.7}})
        self.slam = SlamState(pose=Pose3D(position=(1.0, 1.0, 1.0)))
        self.robot = DroneState(pose=self.slam.pose)

    def test_schema_validation(self):
        output = validate_vlm_output(valid_vlm_json())
        self.assertEqual(output.recommended_subgoal_id, 1)
        self.assertEqual(output.frontier_scores[0].frontier_id, 1)

    def test_heuristic_reasoner_output(self):
        prior = SemanticHeuristicReasoner().infer(
            sensor_packet=self.sensor,
            slam_state=self.slam,
            exploration_map=self.map,
            frontiers=[self.frontier],
            robot_state=self.robot,
        )
        self.assertEqual(prior.recommended_subgoal_id, 1)
        self.assertGreater(prior.frontier_scores[0].score, 0.0)

    def test_vlm_reasoner_mock_and_cache(self):
        calls = {"count": 0}

        def provider(_packet):
            calls["count"] += 1
            return valid_vlm_json()

        with tempfile.TemporaryDirectory() as tmpdir:
            reasoner = VlmSemanticReasoner(provider=provider, cache_dir=tmpdir)
            for _ in range(2):
                prior = reasoner.infer(
                    sensor_packet=self.sensor,
                    slam_state=self.slam,
                    exploration_map=self.map,
                    frontiers=[self.frontier],
                    robot_state=self.robot,
                )
                self.assertEqual(prior.recommended_subgoal_id, 1)
            self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
