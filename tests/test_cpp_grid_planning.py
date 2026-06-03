from __future__ import annotations

import unittest

from exploration_stack.cpp_accel import grid_planning


class GridPlanningFallbackTest(unittest.TestCase):
    def test_astar_and_frontier_outputs(self):
        occupancy = [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1],
        ]
        path, cost = grid_planning.astar_grid((0, 0), (2, 3), occupancy)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 3))
        self.assertGreater(cost, 0.0)
        self.assertNotIn((1, 1), path)

        frontier = grid_planning.extract_frontiers(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
            ],
            0,
            1,
            2,
        )
        self.assertTrue(frontier[1][1])

    def test_coverage_and_components(self):
        visited = [[False, False], [True, False]]
        observed = [[True, False], [True, True]]
        self.assertEqual(grid_planning.update_coverage_bitset(visited, observed), 2)
        self.assertEqual(visited, [[True, False], [True, True]])
        self.assertAlmostEqual(grid_planning.compute_coverage_ratio(visited, [[True, True], [True, False]]), 2 / 3)
        labels, count = grid_planning.connected_components([[True, False, True], [True, False, True]])
        self.assertEqual(count, 2)
        self.assertEqual(labels[0][0], labels[1][0])
        self.assertNotEqual(labels[0][0], labels[0][2])


if __name__ == "__main__":
    unittest.main()
