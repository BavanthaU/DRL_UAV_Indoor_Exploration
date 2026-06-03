from __future__ import annotations

import unittest

from exploration_stack.planning.astar import astar_path, path_cost


class AstarTest(unittest.TestCase):
    def test_path_around_obstacle(self):
        grid = [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1],
        ]
        path = astar_path(grid, (0, 0), (2, 3))
        self.assertIsNotNone(path)
        self.assertNotIn((1, 1), path)
        self.assertGreater(path_cost(path), 0.0)

    def test_blocked_path_returns_none(self):
        grid = [
            [1, 2, 1],
            [1, 2, 1],
            [1, 2, 1],
        ]
        self.assertIsNone(astar_path(grid, (0, 0), (0, 2)))


if __name__ == "__main__":
    unittest.main()
