from __future__ import annotations

import heapq
from collections import deque
from typing import Iterable


try:
    from grid_planning_ext import (  # type: ignore
        astar_grid as _cpp_astar_grid,
        astar_to_any_goal as _cpp_astar_to_any_goal,
        cluster_frontiers as _cpp_cluster_frontiers,
        connected_components as _cpp_connected_components,
        extract_frontiers as _cpp_extract_frontiers,
    )

    CPP_ACCEL_ACTIVE = True
except ImportError:
    CPP_ACCEL_ACTIVE = False
    _cpp_astar_grid = None
    _cpp_astar_to_any_goal = None
    _cpp_cluster_frontiers = None
    _cpp_connected_components = None
    _cpp_extract_frontiers = None


def astar_grid(start, goal, occupancy, cost_map=None, allow_diagonal=False):
    if CPP_ACCEL_ACTIVE:
        return _cpp_astar_grid(tuple(start), tuple(goal), occupancy, cost_map, allow_diagonal)
    return _py_astar_grid(start, goal, occupancy, cost_map=cost_map, allow_diagonal=allow_diagonal)


def astar_to_any_goal(start, goal_mask, occupancy, cost_map=None):
    if CPP_ACCEL_ACTIVE:
        return _cpp_astar_to_any_goal(tuple(start), goal_mask, occupancy, cost_map)
    goals = [(r, c) for r, row in enumerate(goal_mask) for c, value in enumerate(row) if value]
    best_path, best_goal, best_cost = None, None, float("inf")
    for goal in goals:
        path, cost = _py_astar_grid(start, goal, occupancy, cost_map=cost_map)
        if path and cost < best_cost:
            best_path, best_goal, best_cost = path, goal, cost
    return best_path or [], best_goal, best_cost


def extract_frontiers(occupancy_grid, unknown_value=0, free_value=1, occupied_value=2):
    if CPP_ACCEL_ACTIVE:
        return _cpp_extract_frontiers(occupancy_grid, unknown_value, free_value, occupied_value)
    rows, cols = len(occupancy_grid), len(occupancy_grid[0])
    mask = [[False for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            if occupancy_grid[row][col] != free_value:
                continue
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + d_row, col + d_col
                if 0 <= nr < rows and 0 <= nc < cols and occupancy_grid[nr][nc] == unknown_value:
                    mask[row][col] = True
                    break
    return mask


def cluster_frontiers(frontier_mask, min_cluster_size=5):
    if CPP_ACCEL_ACTIVE:
        return _cpp_cluster_frontiers(frontier_mask, min_cluster_size)
    rows, cols = len(frontier_mask), len(frontier_mask[0])
    remaining = {(r, c) for r in range(rows) for c in range(cols) if frontier_mask[r][c]}
    clusters = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        cluster = [start]
        while queue:
            row, col = queue.popleft()
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    candidate = (row + d_row, col + d_col)
                    if candidate in remaining:
                        remaining.remove(candidate)
                        queue.append(candidate)
                        cluster.append(candidate)
        if len(cluster) >= min_cluster_size:
            clusters.append(sorted(cluster))
    return clusters


def update_observed_bitset(visited_bitset, newly_observed_mask):
    count = 0
    for row in range(len(visited_bitset)):
        for col in range(len(visited_bitset[0])):
            if newly_observed_mask[row][col] and not visited_bitset[row][col]:
                visited_bitset[row][col] = True
                count += 1
    return count


def compute_observed_fraction(visited_bitset, candidate_mask):
    valid = 0
    visited = 0
    for row in range(len(candidate_mask)):
        for col in range(len(candidate_mask[0])):
            if candidate_mask[row][col]:
                valid += 1
                visited += bool(visited_bitset[row][col])
    return float(visited / valid) if valid else 0.0


def connected_components(binary_grid):
    if CPP_ACCEL_ACTIVE:
        return _cpp_connected_components(binary_grid)
    rows, cols = len(binary_grid), len(binary_grid[0])
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    label = 0
    for row in range(rows):
        for col in range(cols):
            if not binary_grid[row][col] or labels[row][col] != 0:
                continue
            label += 1
            queue = deque([(row, col)])
            labels[row][col] = label
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and binary_grid[nr][nc] and labels[nr][nc] == 0:
                        labels[nr][nc] = label
                        queue.append((nr, nc))
    return labels, label


def _py_astar_grid(start, goal, occupancy, cost_map=None, allow_diagonal=False):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = len(occupancy), len(occupancy[0])
    heap = [(0.0, tuple(start))]
    came_from = {}
    cost_so_far = {tuple(start): 0.0}
    goal = tuple(goal)
    while heap:
        _, current = heapq.heappop(heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, cost_so_far[goal]
        for dr, dc in offsets:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if occupancy[nr][nc] == 2:
                continue
            step_cost = (2**0.5 if dr and dc else 1.0)
            if cost_map is not None:
                step_cost *= float(cost_map[nr][nc])
            new_cost = cost_so_far[current] + step_cost
            nxt = (nr, nc)
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                priority = new_cost + abs(goal[0] - nr) + abs(goal[1] - nc)
                came_from[nxt] = current
                heapq.heappush(heap, (priority, nxt))
    return [], float("inf")
