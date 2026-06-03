from __future__ import annotations

from collections import deque
from typing import Iterable

from exploration_stack.core import DoorwayHypothesis, Frontier, OccupancyGrid2D, RoomGraph

UNKNOWN = OccupancyGrid2D.UNKNOWN
FREE = OccupancyGrid2D.FREE
OCCUPIED = OccupancyGrid2D.OCCUPIED


def _to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_grid(grid: OccupancyGrid2D | Iterable[Iterable[int]]) -> list[list[int]]:
    if isinstance(grid, OccupancyGrid2D):
        return [[int(value) for value in row] for row in grid.cells]
    return [[int(value) for value in row] for row in _to_list(grid)]


def normalize_occupancy_grid(grid: Iterable[Iterable[int]], source: str = "existing") -> list[list[int]]:
    """Normalize common occupancy formats to 0 unknown, 1 free, 2 occupied."""

    arr = _as_grid(grid)
    normalized = [[UNKNOWN for _ in row] for row in arr]
    if source == "existing":
        for row_index, row in enumerate(arr):
            for col_index, value in enumerate(row):
                if value in (1, 3):
                    normalized[row_index][col_index] = FREE
                elif value == 2:
                    normalized[row_index][col_index] = OCCUPIED
    elif source == "ros":
        for row_index, row in enumerate(arr):
            for col_index, value in enumerate(row):
                if value < 0:
                    normalized[row_index][col_index] = UNKNOWN
                elif value < 50:
                    normalized[row_index][col_index] = FREE
                else:
                    normalized[row_index][col_index] = OCCUPIED
    else:
        raise ValueError(f"Unknown occupancy source '{source}'. Use 'existing' or 'ros'.")
    return normalized


def neighbor_offsets(connectivity: int = 4) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError("connectivity must be 4 or 8")


def extract_frontier_cells(
    occupancy_grid: OccupancyGrid2D | Iterable[Iterable[int]],
    *,
    connectivity: int = 4,
) -> list[tuple[int, int]]:
    """Return free cells adjacent to unknown cells."""

    arr = _as_grid(occupancy_grid)
    if not arr or not arr[0]:
        raise ValueError("occupancy_grid must be a 2D grid")

    rows, cols = len(arr), len(arr[0])
    offsets = neighbor_offsets(connectivity)
    frontiers: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if arr[row][col] != FREE:
                continue
            for d_row, d_col in offsets:
                n_row, n_col = row + d_row, col + d_col
                if 0 <= n_row < rows and 0 <= n_col < cols and arr[n_row][n_col] == UNKNOWN:
                    frontiers.append((row, col))
                    break
    return frontiers


def cluster_frontiers(
    frontier_cells: Iterable[tuple[int, int]],
    *,
    min_cluster_size: int = 5,
    connectivity: int = 8,
) -> list[list[tuple[int, int]]]:
    """Cluster frontier cells with connected components."""

    cells = set(frontier_cells)
    offsets = neighbor_offsets(connectivity)
    clusters: list[list[tuple[int, int]]] = []
    while cells:
        start = cells.pop()
        queue: deque[tuple[int, int]] = deque([start])
        cluster = [start]
        while queue:
            row, col = queue.popleft()
            for d_row, d_col in offsets:
                candidate = (row + d_row, col + d_col)
                if candidate in cells:
                    cells.remove(candidate)
                    queue.append(candidate)
                    cluster.append(candidate)
        if len(cluster) >= min_cluster_size:
            clusters.append(sorted(cluster))
    return clusters


def frontiers_from_occupancy_grid(
    occupancy_grid: OccupancyGrid2D | Iterable[Iterable[int]],
    *,
    min_cluster_size: int = 5,
    connectivity: int = 8,
    start_id: int = 0,
) -> list[Frontier]:
    """Extract clustered frontiers and compute map-frame centroids."""

    if isinstance(occupancy_grid, OccupancyGrid2D):
        grid = occupancy_grid
    else:
        grid = OccupancyGrid2D(cells=_as_grid(occupancy_grid))

    frontier_cells = extract_frontier_cells(grid, connectivity=4)
    clusters = cluster_frontiers(frontier_cells, min_cluster_size=min_cluster_size, connectivity=connectivity)
    frontiers: list[Frontier] = []
    for index, cluster in enumerate(clusters):
        row_mean = sum(cell[0] for cell in cluster) / len(cluster)
        col_mean = sum(cell[1] for cell in cluster) / len(cluster)
        centroid_xy = grid.cell_to_xy((int(round(row_mean)), int(round(col_mean))))
        frontiers.append(
            Frontier(
                frontier_id=start_id + index,
                cells=cluster,
                centroid_xy=centroid_xy,
                information_gain=float(len(cluster)),
                size=len(cluster),
            )
        )
    return frontiers


def coverage_fraction(occupancy_grid: OccupancyGrid2D | Iterable[Iterable[int]]) -> float:
    """Return the fraction of cells that are no longer unknown."""

    arr = _as_grid(occupancy_grid)
    total = sum(len(row) for row in arr)
    if total == 0:
        return 0.0
    known = sum(1 for row in arr for value in row if value != UNKNOWN)
    return known / total


def occupancy3d_to_2d(occupancy_3d) -> list[list[int]]:
    """Project a 3D occupancy volume to a 2D exploration grid."""

    arr = _to_list(occupancy_3d)
    if not arr or not isinstance(arr[0], list) or not isinstance(arr[0][0], list):
        raise ValueError("occupancy_3d must have shape (height, rows, cols)")
    rows = len(arr[0])
    cols = len(arr[0][0])
    result = [[UNKNOWN for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            values = [int(layer[row][col]) for layer in arr]
            if any(value == OCCUPIED for value in values):
                result[row][col] = OCCUPIED
            elif any(value != UNKNOWN for value in values):
                result[row][col] = FREE
    return result


def esdf_to_safe_occupancy_2d(
    esdf_distances_m: Iterable[Iterable[float]],
    *,
    safety_radius_m: float,
) -> list[list[int]]:
    """Convert ESDF distances into free/occupied cells under a safety radius."""

    import math

    distances = [[float(value) for value in row] for row in _to_list(esdf_distances_m)]
    result = [[FREE for _ in row] for row in distances]
    for row_index, row in enumerate(distances):
        for col_index, distance in enumerate(row):
            if not math.isfinite(distance):
                result[row_index][col_index] = UNKNOWN
            elif distance < safety_radius_m:
                result[row_index][col_index] = OCCUPIED
    return result


def update_room_graph_from_doorways(
    room_graph: RoomGraph,
    doorways: Iterable[DoorwayHypothesis],
    *,
    explored_region_id: int | None = None,
) -> RoomGraph:
    """Update a lightweight room graph with doorway hypotheses."""

    next_rooms = dict(room_graph.rooms)
    next_edges = list(room_graph.edges)
    next_doorways = list(room_graph.doorways)
    for doorway in doorways:
        if all(existing.doorway_id != doorway.doorway_id for existing in next_doorways):
            next_doorways.append(doorway)
        if explored_region_id is not None:
            next_rooms.setdefault(explored_region_id, {"source": "connected_explored_region"})
    return RoomGraph(rooms=next_rooms, edges=next_edges, doorways=next_doorways)
