from __future__ import annotations

import unittest

from exploration_stack.core import OccupancyGrid2D
from exploration_stack.mapping.frontier_utils import (
    coverage_fraction,
    extract_frontier_cells,
    frontiers_from_occupancy_grid,
    normalize_occupancy_grid,
    occupancy3d_to_2d,
)


class FrontierUtilsTest(unittest.TestCase):
    def test_existing_occupancy_normalization(self):
        grid = normalize_occupancy_grid([[0, 1, 2, 3]], source="existing")
        self.assertEqual(grid, [[0, 1, 2, 1]])

    def test_frontier_extraction_and_clustering(self):
        grid = OccupancyGrid2D(
            cells=[
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            resolution_m=1.0,
        )
        cells = extract_frontier_cells(grid)
        self.assertIn((1, 1), cells)
        self.assertNotIn((2, 2), cells)
        frontiers = frontiers_from_occupancy_grid(grid, min_cluster_size=2)
        self.assertGreaterEqual(len(frontiers), 1)
        self.assertGreater(frontiers[0].information_gain, 0.0)

    def test_coverage_and_3d_projection(self):
        grid = [[0, 1], [2, 1]]
        self.assertAlmostEqual(coverage_fraction(grid), 0.75)
        volume = [[[0, 1], [0, 0]], [[0, 0], [2, 1]]]
        projected = occupancy3d_to_2d(volume)
        self.assertEqual(projected, [[0, 1], [2, 1]])


if __name__ == "__main__":
    unittest.main()
