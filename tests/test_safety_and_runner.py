from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exploration_stack.control.safety_shield import EsdfSafetyShield
from exploration_stack.core import EsdfMap, ExplorationMap, LocalCommand, OccupancyGrid2D, Pose3D, SlamState
from run_four_layer_smoke_test import build_mock_runner


class SafetyAndRunnerTest(unittest.TestCase):
    def test_safety_shield_reduces_forward_velocity(self):
        shield = EsdfSafetyShield(min_clearance_m=0.45, slow_forward_scale=0.25)
        grid = OccupancyGrid2D(cells=[[1, 1], [1, 1]], resolution_m=1.0)
        exploration_map = ExplorationMap(occupancy_2d=grid, esdf=EsdfMap(distances_m=[[0.2, 1.0], [1.0, 1.0]]))
        command = LocalCommand(velocity_body=(1.0, 0.0, 0.0))
        filtered = shield.filter(command, SlamState(pose=Pose3D(position=(0.1, 0.1, 0.0))), exploration_map)
        self.assertLess(filtered.velocity_body[0], command.velocity_body[0])
        self.assertEqual(filtered.metadata["safety_intervention"], "reduced_forward_velocity")

    def test_full_four_layer_runner_with_mocks(self):
        runner = build_mock_runner()
        result = runner.run_episode(max_steps=4)
        self.assertGreater(result.steps, 0)
        self.assertGreater(result.coverage, 0.0)
        self.assertIsNotNone(result.subgoal)
        self.assertGreaterEqual(result.records[0]["num_frontiers"], 1)


if __name__ == "__main__":
    unittest.main()
