from __future__ import annotations

import heapq
import math
from typing import Iterable

from exploration_stack.core import OccupancyGrid2D
from exploration_stack.mapping.frontier_utils import FREE, OCCUPIED, UNKNOWN, neighbor_offsets


def _as_grid(grid: OccupancyGrid2D | Iterable[Iterable[int]]) -> list[list[int]]:
    if isinstance(grid, OccupancyGrid2D):
        return [[int(value) for value in row] for row in grid.cells]
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [[int(value) for value in row] for row in grid]


def inflate_obstacles(grid: list[list[int]], radius_cells: int) -> list[list[int]]:
    if radius_cells <= 0:
        return [row[:] for row in grid]
    inflated = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0]) if grid else 0
    obstacle_cells = [(row, col) for row in range(height) for col in range(width) if grid[row][col] == OCCUPIED]
    for row, col in obstacle_cells:
        for d_row in range(-radius_cells, radius_cells + 1):
            for d_col in range(-radius_cells, radius_cells + 1):
                if math.hypot(d_row, d_col) > radius_cells:
                    continue
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < height and 0 <= n_col < width:
                    inflated[n_row][n_col] = OCCUPIED
    return inflated


def astar_path(
    occupancy_grid: OccupancyGrid2D | Iterable[Iterable[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    allow_unknown: bool = False,
    safety_radius_cells: int = 0,
    connectivity: int = 4,
) -> list[tuple[int, int]] | None:
    """Compute a grid path from start to goal using A*."""

    grid = inflate_obstacles(_as_grid(occupancy_grid), safety_radius_cells)
    if not grid or not grid[0]:
        raise ValueError("occupancy_grid must be 2D")
    if not _in_bounds(grid, start) or not _in_bounds(grid, goal):
        return None
    if not _is_traversable(grid, start, allow_unknown) or not _is_traversable(grid, goal, allow_unknown):
        return None

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    offsets = neighbor_offsets(connectivity)

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current)
        closed.add(current)

        for d_row, d_col in offsets:
            neighbor = (current[0] + d_row, current[1] + d_col)
            if not _in_bounds(grid, neighbor) or not _is_traversable(grid, neighbor, allow_unknown):
                continue
            step_cost = math.hypot(d_row, d_col)
            tentative = g_score[current] + step_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, neighbor))
    return None


def path_cost(path: list[tuple[int, int]] | None, *, resolution_m: float = 1.0) -> float:
    if not path:
        return float("inf")
    cost = 0.0
    for previous, current in zip(path[:-1], path[1:]):
        cost += math.hypot(current[0] - previous[0], current[1] - previous[1]) * resolution_m
    return cost


def _in_bounds(grid: list[list[int]], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < len(grid) and 0 <= cell[1] < len(grid[0])


def _is_traversable(grid: list[list[int]], cell: tuple[int, int], allow_unknown: bool) -> bool:
    value = grid[cell[0]][cell[1]]
    if value == OCCUPIED:
        return False
    if value == UNKNOWN and not allow_unknown:
        return False
    return value == FREE or (allow_unknown and value == UNKNOWN)


def _heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> float:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
