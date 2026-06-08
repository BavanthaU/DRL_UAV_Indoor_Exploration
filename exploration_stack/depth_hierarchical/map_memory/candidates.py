from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import atan2, cos, hypot, pi, sin

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.cpp_accel import grid_planning

from .values import FREE, OCCUPIED, UNKNOWN, VISITED

MAX_CANDIDATES = 64


class CandidateType(IntEnum):
    FRONTIER_CLUSTER = 0
    OPEN_END = 1
    DOOR_LIKE_OPENING_GEOMETRIC = 2
    CORRIDOR_CONTINUATION_GEOMETRIC = 3
    MISSING_PATCH = 4
    BACKTRACK_NODE = 5


NUM_CANDIDATE_TYPES = len(CandidateType)
CANDIDATE_FEATURE_DIM = NUM_CANDIDATE_TYPES + 18


@dataclass
class Candidate:
    candidate_id: int
    type: CandidateType
    row: int
    col: int
    cluster_size: int = 1
    age: int = 0
    failed_attempt_count: int = 0
    last_selected_time: float = -1.0


@dataclass
class MapMemoryConfig:
    height: int = 64
    width: int = 64
    resolution_m: float = 0.10
    max_candidates: int = MAX_CANDIDATES
    candidate_radius_cells: int = 5
    min_frontier_cluster_size: int = 3
    raycast_max_cells: int = 18
    candidate_feature_dim: int = CANDIDATE_FEATURE_DIM


@dataclass
class MapMemory:
    cfg: MapMemoryConfig = field(default_factory=MapMemoryConfig)

    def __post_init__(self):
        if np is None:
            raise RuntimeError("MapMemory requires numpy.")
        self.grid = np.full((self.cfg.height, self.cfg.width), UNKNOWN, dtype=np.int16)
        self.visited = np.zeros((self.cfg.height, self.cfg.width), dtype=bool)
        self._candidate_age: dict[tuple[int, int, int], int] = {}
        self._failed: dict[tuple[int, int, int], int] = {}
        self._last_selected: dict[tuple[int, int, int], float] = {}
        self.time = 0.0

    def mark_observed(self, free_cells=(), occupied_cells=(), agent_cell=None):
        newly_known = 0
        for row, col in free_cells:
            if self._inside(row, col):
                newly_known += int(self.grid[row, col] == UNKNOWN)
                self.grid[row, col] = FREE
        for row, col in occupied_cells:
            if self._inside(row, col):
                newly_known += int(self.grid[row, col] == UNKNOWN)
                self.grid[row, col] = OCCUPIED
        if agent_cell is not None and self._inside(*agent_cell):
            self.grid[agent_cell] = VISITED
            self.visited[agent_cell] = True
        return newly_known

    def extract_candidates(self, agent_cell: tuple[int, int], yaw_rad: float = 0.0) -> list[Candidate]:
        candidates: list[Candidate] = []
        candidates.extend(self._frontier_cluster_candidates())
        candidates.extend(self._open_end_candidates())
        candidates.extend(self._missing_patch_candidates())
        candidates.extend(self._backtrack_candidates(agent_cell))
        unique: dict[tuple[int, int, int], Candidate] = {}
        for candidate in candidates:
            key = (int(candidate.type), candidate.row, candidate.col)
            if key in unique:
                unique[key].cluster_size += candidate.cluster_size
            else:
                age = self._candidate_age.get(key, 0) + 1
                self._candidate_age[key] = age
                candidate.age = age
                candidate.failed_attempt_count = self._failed.get(key, 0)
                candidate.last_selected_time = self._last_selected.get(key, -1.0)
                unique[key] = candidate
        ordered = list(unique.values())
        ordered.sort(key=lambda c: (int(c.type), c.candidate_id))
        return ordered[: self.cfg.max_candidates]

    def candidate_tensors(self, agent_cell: tuple[int, int], yaw_rad: float = 0.0, device="cpu"):
        if torch is None:
            raise RuntimeError("candidate_tensors requires PyTorch.")
        candidates = self.extract_candidates(agent_cell, yaw_rad=yaw_rad)
        features = torch.zeros(1, self.cfg.max_candidates, self.cfg.candidate_feature_dim, device=device)
        mask = torch.zeros(1, self.cfg.max_candidates, dtype=torch.bool, device=device)
        for idx, candidate in enumerate(candidates[: self.cfg.max_candidates]):
            features[0, idx] = torch.as_tensor(self.candidate_features(candidate, agent_cell, yaw_rad), device=device)
            mask[0, idx] = True
        return features, mask, candidates

    def candidate_features(self, candidate: Candidate, agent_cell: tuple[int, int], yaw_rad: float = 0.0):
        feature = np.zeros(self.cfg.candidate_feature_dim, dtype=np.float32)
        feature[int(candidate.type)] = 1.0
        dr = float(candidate.row - agent_cell[0])
        dc = float(candidate.col - agent_cell[1])
        distance_cells = hypot(dr, dc)
        path, path_length = grid_planning.astar_grid(agent_cell, (candidate.row, candidate.col), self._astar_occupancy())
        astar_reachable = bool(path)
        counts = self._count_radius(candidate.row, candidate.col, self.cfg.candidate_radius_cells)
        open_angle, narrowness = self._local_open_geometry(candidate.row, candidate.col)
        clearance = self._clearance(candidate.row, candidate.col)
        seen = self._already_seen_ratio(candidate.row, candidate.col)
        expected = grid_planning.raycast_unknown_gain(
            self.grid.tolist(),
            (candidate.row, candidate.col),
            yaw_rad,
            max_range_cells=self.cfg.raycast_max_cells,
            unknown_value=UNKNOWN,
            occupied_value=OCCUPIED,
        )
        offset = NUM_CANDIDATE_TYPES
        values = [
            dc * self.cfg.resolution_m,
            dr * self.cfg.resolution_m,
            distance_cells * self.cfg.resolution_m,
            atan2(dr, dc) / pi,
            float(path_length),
            float(astar_reachable),
            float(candidate.cluster_size),
            float(counts[UNKNOWN]),
            float(counts[FREE] + counts[VISITED]),
            float(counts[OCCUPIED]),
            float(open_angle),
            float(narrowness),
            float(clearance),
            float(candidate.age),
            float(candidate.failed_attempt_count),
            float(candidate.last_selected_time),
            float(seen),
            float(expected),
        ]
        feature[offset : offset + len(values)] = np.asarray(values[: self.cfg.candidate_feature_dim - offset])
        return feature

    def global_stats(self):
        total = float(self.grid.size)
        return np.asarray(
            [
                np.count_nonzero(self.grid == UNKNOWN) / total,
                np.count_nonzero(self.grid == FREE) / total,
                np.count_nonzero(self.grid == OCCUPIED) / total,
                np.count_nonzero(self.grid == VISITED) / total,
                np.count_nonzero(self.visited) / total,
                float(self.cfg.resolution_m),
                float(self.cfg.height),
                float(self.cfg.width),
            ],
            dtype=np.float32,
        )

    def map_crop_tensor(self, device="cpu"):
        if torch is None:
            raise RuntimeError("map_crop_tensor requires PyTorch.")
        channels = np.stack(
            [
                self.grid == UNKNOWN,
                self.grid == FREE,
                self.grid == OCCUPIED,
                (self.grid == VISITED) | self.visited,
            ],
            axis=0,
        ).astype(np.float32)
        return torch.as_tensor(channels[None], device=device)

    def record_selected(self, candidate: Candidate):
        key = (int(candidate.type), candidate.row, candidate.col)
        self._last_selected[key] = self.time

    def record_failed(self, candidate: Candidate):
        key = (int(candidate.type), candidate.row, candidate.col)
        self._failed[key] = self._failed.get(key, 0) + 1

    def _frontier_cluster_candidates(self):
        frontier = grid_planning.extract_frontiers(self.grid.tolist(), UNKNOWN, FREE, OCCUPIED)
        clusters = grid_planning.cluster_frontiers(frontier, self.cfg.min_frontier_cluster_size)
        out = []
        for idx, cluster in enumerate(clusters):
            row = int(round(sum(cell[0] for cell in cluster) / len(cluster)))
            col = int(round(sum(cell[1] for cell in cluster) / len(cluster)))
            out.append(Candidate(idx, CandidateType.FRONTIER_CLUSTER, row, col, cluster_size=len(cluster)))
        return out

    def _open_end_candidates(self):
        out = []
        idx = 1000
        for row, col in zip(*np.nonzero((self.grid == FREE) | (self.grid == VISITED))):
            unknown_neighbors = self._neighbor_count(row, col, UNKNOWN)
            free_neighbors = self._neighbor_count(row, col, FREE) + self._neighbor_count(row, col, VISITED)
            if unknown_neighbors >= 2 and free_neighbors <= 3:
                out.append(Candidate(idx, CandidateType.OPEN_END, int(row), int(col)))
                idx += 1
            if unknown_neighbors >= 2 and free_neighbors == 2:
                out.append(Candidate(idx, CandidateType.DOOR_LIKE_OPENING_GEOMETRIC, int(row), int(col)))
                idx += 1
            if unknown_neighbors >= 1 and self._looks_like_corridor(row, col):
                out.append(Candidate(idx, CandidateType.CORRIDOR_CONTINUATION_GEOMETRIC, int(row), int(col)))
                idx += 1
        return out

    def _missing_patch_candidates(self):
        out = []
        idx = 3000
        for row, col in zip(*np.nonzero(self.grid == UNKNOWN)):
            known = self._neighbor_count(row, col, FREE) + self._neighbor_count(row, col, VISITED)
            if known >= 5:
                out.append(Candidate(idx, CandidateType.MISSING_PATCH, int(row), int(col)))
                idx += 1
        return out

    def _backtrack_candidates(self, agent_cell):
        out = []
        rows, cols = np.nonzero(self.visited)
        for idx, (row, col) in enumerate(zip(rows, cols)):
            if abs(int(row) - agent_cell[0]) + abs(int(col) - agent_cell[1]) >= 5:
                out.append(Candidate(5000 + idx, CandidateType.BACKTRACK_NODE, int(row), int(col)))
        return out

    def _astar_occupancy(self):
        occ = np.zeros_like(self.grid, dtype=np.int16)
        occ[self.grid == OCCUPIED] = 2
        return occ.tolist()

    def _count_radius(self, row, col, radius):
        counts = {UNKNOWN: 0, FREE: 0, OCCUPIED: 0, VISITED: 0}
        r0, r1 = max(0, row - radius), min(self.cfg.height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(self.cfg.width, col + radius + 1)
        patch = self.grid[r0:r1, c0:c1]
        for value in counts:
            counts[value] = int(np.count_nonzero(patch == value))
        return counts

    def _local_open_geometry(self, row, col):
        free_dirs = 0
        occupied_dirs = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if self._inside(nr, nc) and self.grid[nr, nc] in (FREE, VISITED, UNKNOWN):
                free_dirs += 1
            if self._inside(nr, nc) and self.grid[nr, nc] == OCCUPIED:
                occupied_dirs += 1
        return free_dirs / 8.0, occupied_dirs / 8.0

    def _clearance(self, row, col):
        occupied = np.argwhere(self.grid == OCCUPIED)
        if occupied.size == 0:
            return float(self.cfg.raycast_max_cells * self.cfg.resolution_m)
        dist = np.sqrt(((occupied - np.asarray([row, col])) ** 2).sum(axis=1)).min()
        return float(dist * self.cfg.resolution_m)

    def _already_seen_ratio(self, row, col):
        counts = self._count_radius(row, col, self.cfg.candidate_radius_cells)
        total = max(1, sum(counts.values()))
        return float((counts[FREE] + counts[OCCUPIED] + counts[VISITED]) / total)

    def _neighbor_count(self, row, col, value):
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(row) + dr, int(col) + dc
                count += int(self._inside(nr, nc) and self.grid[nr, nc] == value)
        return count

    def _looks_like_corridor(self, row, col):
        horizontal = int(self._inside(row, col - 1) and self.grid[row, col - 1] in (FREE, VISITED, UNKNOWN))
        horizontal += int(self._inside(row, col + 1) and self.grid[row, col + 1] in (FREE, VISITED, UNKNOWN))
        vertical = int(self._inside(row - 1, col) and self.grid[row - 1, col] in (FREE, VISITED, UNKNOWN))
        vertical += int(self._inside(row + 1, col) and self.grid[row + 1, col] in (FREE, VISITED, UNKNOWN))
        return (horizontal >= 2 and vertical <= 1) or (vertical >= 2 and horizontal <= 1)

    def _inside(self, row, col):
        return 0 <= int(row) < self.cfg.height and 0 <= int(col) < self.cfg.width
