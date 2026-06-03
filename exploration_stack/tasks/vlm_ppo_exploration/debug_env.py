from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

from exploration_stack.cpp_accel import grid_planning
from exploration_stack.vlm_frontend.map_renderer import MapRenderConfig, MapRenderer

from .intrinsic_rewards import NewCellCountCuriosity
from .observations import crop_grid
from .reward_terms import RewardWeights, combine_rewards, compute_extrinsic_reward


@dataclass
class DebugVlmPpoEnvConfig:
    num_envs: int = 2
    grid_size: int = 32
    crop_size: int = 16
    image_size: int = 64
    max_steps: int = 64
    sensor_radius_cells: int = 3
    max_vx_cells: float = 1.0
    max_vy_cells: float = 0.8
    max_yaw_rate: float = 0.35
    stuck_steps: int = 24
    frontier_closed_steps: int = 6
    min_mapped_cells_for_completion: int = 16
    return_home_fraction: float = 0.8
    home_reached_radius_cells: float = 2.0
    device: str = "cpu"
    seed: int = 7


class DebugVlmPpoVectorEnv:
    """CPU vector environment for deterministic PPO smoke training.

    The real task is the Isaac Lab DirectRLEnv. This environment keeps the
    same observation/action contract and uses map dynamics, frontier extraction,
    A* subgoal features, collisions, and new-cell rewards for fast tests.
    """

    def __init__(self, cfg: DebugVlmPpoEnvConfig | None = None):
        if torch is None:
            raise RuntimeError("DebugVlmPpoVectorEnv requires PyTorch.")
        self.cfg = cfg or DebugVlmPpoEnvConfig()
        self.device = torch.device(self.cfg.device)
        self.num_envs = self.cfg.num_envs
        self.action_dim = 3
        self._generator = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        shape = (self.num_envs, self.cfg.grid_size, self.cfg.grid_size)
        self._true_map = torch.ones(shape, dtype=torch.long, device=self.device)
        self._known_map = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._trajectory = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._pose = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mapped_free_cells = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_mapped_free_cells = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._frontier_closed_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32, device=self.device)
        self._home_pose = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self._prev_home_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._curiosity = NewCellCountCuriosity(
            self.num_envs,
            (self.cfg.grid_size, self.cfg.grid_size),
            device=self.device,
        )
        self._weights = RewardWeights()
        self._renderer = MapRenderer(MapRenderConfig(image_size=self.cfg.image_size))
        self.reset()

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        for env_id in ids.tolist():
            self._true_map[env_id] = self._build_map(env_id)
        center = self.cfg.grid_size // 2
        self._known_map[ids] = 0
        self._trajectory[ids] = False
        self._pose[ids] = torch.tensor([center, center], dtype=torch.float32, device=self.device)
        self._home_pose[ids] = self._pose[ids]
        self._yaw[ids] = 0.0
        self._steps[ids] = 0
        self._stuck_counter[ids] = 0
        self._mapped_free_cells[ids] = 0.0
        self._prev_mapped_free_cells[ids] = 0.0
        self._frontier_closed_counter[ids] = 0
        self._prev_home_distance[ids] = 0.0
        self._prev_actions[ids] = 0.0
        self._curiosity.reset(ids)
        self._reveal(ids)
        self._mapped_free_cells[ids] = (self._known_map[ids] == 1).flatten(start_dim=1).sum(dim=1).float()
        self._prev_mapped_free_cells[ids] = self._mapped_free_cells[ids]
        self._curiosity.prime(self._known_map, ids)
        return self._get_obs()

    def step(self, actions):
        actions = actions.to(self.device).float().clamp(-1.0, 1.0)
        self._steps += 1
        prev_mapped_free_cells = self._mapped_free_cells.clone()
        collision = self._move(actions)
        self._reveal()
        occupancy_for_reward = self._known_map.clone()
        new_cell_reward, new_counts = self._curiosity.update(occupancy_for_reward)
        frontier = self._frontier_mask()
        frontier_count = frontier.flatten(start_dim=1).sum(dim=1).float()
        self._mapped_free_cells = (self._known_map == 1).flatten(start_dim=1).sum(dim=1).float()
        mapped_cell_delta = (self._mapped_free_cells - prev_mapped_free_cells).clamp_min(0.0)
        frontier_closed_now = (frontier_count <= 0) & (self._mapped_free_cells >= self.cfg.min_mapped_cells_for_completion)
        self._frontier_closed_counter = torch.where(
            frontier_closed_now,
            self._frontier_closed_counter + 1,
            torch.zeros_like(self._frontier_closed_counter),
        )
        frontier_complete = self._frontier_closed_counter >= self.cfg.frontier_closed_steps
        return_home = self._return_home_mask()
        home_distance = torch.linalg.norm(self._pose - self._home_pose, dim=-1)
        return_home_progress = (self._prev_home_distance - home_distance).clamp_min(0.0)
        return_home_progress = return_home_progress * return_home.float() / max(1.0, float(self.cfg.grid_size))
        low_motion = torch.linalg.norm(actions[:, :2], dim=-1) < 0.05
        idle = low_motion & (new_counts <= 0)
        yaw_flip = (torch.sign(actions[:, 2]) != torch.sign(self._prev_actions[:, 2])) & (actions[:, 2].abs() > 0.1)
        yaw_flip &= self._prev_actions[:, 2].abs() > 0.1
        self._stuck_counter = torch.where(new_counts > 0, torch.zeros_like(self._stuck_counter), self._stuck_counter + 1)
        extrinsic, extrinsic_terms = compute_extrinsic_reward(
            map_progress=new_cell_reward.detach(),
            frontier_closed=frontier_complete,
            collision=collision,
            esdf_clearance=torch.where(collision, torch.zeros_like(new_cell_reward), torch.ones_like(new_cell_reward)),
            safety_radius=0.45,
            dt=1.0,
            idle_mask=idle,
            yaw_flip=yaw_flip,
            action=actions,
            prev_action=self._prev_actions,
            altitude=torch.ones_like(new_cell_reward) * 1.2,
            target_altitude=1.2,
            return_home_progress=return_home_progress,
            weights=self._weights,
        )
        reward, intrinsic_terms = combine_rewards(
            extrinsic,
            new_cell_reward,
            torch.zeros_like(new_cell_reward),
            torch.zeros_like(new_cell_reward),
            self._weights,
        )
        returned_home = return_home & (home_distance <= self.cfg.home_reached_radius_cells)
        done = collision | frontier_complete | returned_home | (self._steps >= self.cfg.max_steps) | (self._stuck_counter >= self.cfg.stuck_steps)
        reward_terms = {
            **extrinsic_terms,
            **intrinsic_terms,
            "new_cells": new_counts,
            "mapped_free_cells": self._mapped_free_cells,
            "mapped_cell_delta": mapped_cell_delta,
            "frontier_count": frontier_count,
            "frontier_closed": frontier_complete.float(),
            "return_home_phase": return_home.float(),
            "home_distance": home_distance,
        }
        reward_terms = {key: value.detach().clone() for key, value in reward_terms.items()}
        self._prev_actions = actions.detach().clone()
        self._prev_mapped_free_cells = self._mapped_free_cells.detach().clone()
        self._prev_home_distance = home_distance.detach().clone()
        if done.any():
            self.reset(torch.nonzero(done, as_tuple=False).flatten())
        return self._get_obs(), reward.detach(), done.detach(), {"reward_terms": reward_terms}

    def _build_map(self, env_id: int):
        size = self.cfg.grid_size
        grid = torch.ones(size, size, dtype=torch.long, device=self.device)
        grid[0, :] = 2
        grid[-1, :] = 2
        grid[:, 0] = 2
        grid[:, -1] = 2
        offset = (env_id * 3) % 7
        mid = size // 2
        grid[5 + offset : size - 5, mid - 4] = 2
        grid[5:size - 5 - offset, mid + 5] = 2
        grid[mid - 4, 4:size - 4] = 2
        grid[mid + 5, 4:size - 4] = 2
        for row, col in ((mid, mid - 4), (mid - 4, mid + 1), (mid + 5, mid - 6), (mid + 1, mid + 5)):
            grid[max(1, row - 1) : min(size - 1, row + 2), max(1, col - 1) : min(size - 1, col + 2)] = 1
        return grid

    def _move(self, actions):
        self._yaw = self._wrap_angle(self._yaw + actions[:, 2] * self.cfg.max_yaw_rate)
        vx = actions[:, 0] * self.cfg.max_vx_cells
        vy = actions[:, 1] * self.cfg.max_vy_cells
        cos_yaw = torch.cos(self._yaw)
        sin_yaw = torch.sin(self._yaw)
        delta_col = cos_yaw * vx - sin_yaw * vy
        delta_row = sin_yaw * vx + cos_yaw * vy
        proposal = self._pose + torch.stack([delta_row, delta_col], dim=-1)
        proposal_idx = proposal.round().long().clamp(0, self.cfg.grid_size - 1)
        collision = self._true_map[torch.arange(self.num_envs, device=self.device), proposal_idx[:, 0], proposal_idx[:, 1]] == 2
        self._pose = torch.where(collision.unsqueeze(-1), self._pose, proposal)
        pose_idx = self._centers()
        self._trajectory[torch.arange(self.num_envs, device=self.device), pose_idx[:, 0], pose_idx[:, 1]] = True
        return collision

    def _reveal(self, env_ids=None):
        ids = self._ids(env_ids)
        radius = self.cfg.sensor_radius_cells
        centers = self._centers()
        for env_id in ids.tolist():
            row = int(centers[env_id, 0].item())
            col = int(centers[env_id, 1].item())
            row0, row1 = max(0, row - radius), min(self.cfg.grid_size, row + radius + 1)
            col0, col1 = max(0, col - radius), min(self.cfg.grid_size, col + radius + 1)
            self._known_map[env_id, row0:row1, col0:col1] = self._true_map[env_id, row0:row1, col0:col1]

    def _get_obs(self):
        centers = self._centers()
        frontier = self._frontier_mask()
        map_crop = crop_grid(self._known_map, centers, self.cfg.crop_size)
        frontier_crop = crop_grid(frontier.long(), centers, self.cfg.crop_size)
        trajectory_crop = crop_grid(self._trajectory.long(), centers, self.cfg.crop_size)
        camera_rgb = self._renderer.render(
            map_crop,
            frontier_mask=frontier_crop,
            trajectory_mask=trajectory_crop,
            robot_xy=torch.full((self.num_envs, 2), self.cfg.crop_size // 2, device=self.device),
        )
        return {
            "camera_rgb": camera_rgb,
            "map_crop": map_crop,
            "frontier_mask": frontier_crop,
            "trajectory_mask": trajectory_crop,
            "depth_line": self._depth_line(),
            "semantic_line": self._semantic_line(),
            "subgoal_features": self._subgoal_features(frontier),
        }

    def _frontier_mask(self):
        frontier = torch.zeros_like(self._known_map, dtype=torch.bool)
        free = self._known_map == 1
        unknown = self._known_map == 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            frontier |= free & torch.roll(unknown, shifts=(dr, dc), dims=(1, 2))
        frontier[:, 0, :] = False
        frontier[:, -1, :] = False
        frontier[:, :, 0] = False
        frontier[:, :, -1] = False
        return frontier

    def _depth_line(self, rays: int = 64, max_range: float = 8.0):
        angles = torch.linspace(-math.pi / 3, math.pi / 3, rays, device=self.device)
        depth = torch.zeros(self.num_envs, rays, dtype=torch.float32, device=self.device)
        for env_id in range(self.num_envs):
            for ray_id, rel_angle in enumerate(angles):
                depth[env_id, ray_id] = self._ray_distance(env_id, self._yaw[env_id] + rel_angle, max_range) / max_range
        return depth

    def _semantic_line(self, rays: int = 64, max_range: float = 8.0):
        angles = torch.linspace(-math.pi / 3, math.pi / 3, rays, device=self.device)
        labels = torch.zeros(self.num_envs, rays, dtype=torch.float32, device=self.device)
        for env_id in range(self.num_envs):
            for ray_id, rel_angle in enumerate(angles):
                distance = self._ray_distance(env_id, self._yaw[env_id] + rel_angle, max_range)
                labels[env_id, ray_id] = 1.0 if distance < max_range else 0.0
        return labels

    def _ray_distance(self, env_id: int, angle, max_range: float):
        direction = torch.stack([torch.sin(angle), torch.cos(angle)])
        origin = self._pose[env_id]
        for step in torch.linspace(0.5, max_range, 16, device=self.device):
            cell = (origin + direction * step).round().long().clamp(0, self.cfg.grid_size - 1)
            if self._true_map[env_id, cell[0], cell[1]] == 2:
                return float(step.item())
        return max_range

    def _subgoal_features(self, frontier):
        features = torch.zeros(self.num_envs, 5, dtype=torch.float32, device=self.device)
        centers = self._centers()
        for env_id in range(self.num_envs):
            if self._return_home_mask()[env_id]:
                delta_grid = self._home_pose[env_id] - self._pose[env_id]
                distance = torch.linalg.norm(delta_grid)
                bearing_world = torch.atan2(delta_grid[0], delta_grid[1])
                bearing_body = self._wrap_angle(bearing_world - self._yaw[env_id])
                features[env_id, 0] = distance / self.cfg.grid_size
                features[env_id, 1] = bearing_body / math.pi
                features[env_id, 3] = 1.0
                features[env_id, 4] = self._stuck_counter[env_id].float() / max(1, self.cfg.stuck_steps)
                continue
            frontier_list = frontier[env_id].cpu().tolist()
            planning_map = self._known_map[env_id].clone()
            planning_map[planning_map == 0] = 2
            path, goal, cost = grid_planning.astar_to_any_goal(
                tuple(centers[env_id].cpu().tolist()),
                frontier_list,
                planning_map.cpu().tolist(),
            )
            if goal is None or not path:
                features[env_id, 4] = self._stuck_counter[env_id].float() / max(1, self.cfg.stuck_steps)
                continue
            goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
            delta_grid = goal_tensor - self._pose[env_id]
            distance = torch.linalg.norm(delta_grid)
            bearing_world = torch.atan2(delta_grid[0], delta_grid[1])
            bearing_body = self._wrap_angle(bearing_world - self._yaw[env_id])
            features[env_id, 0] = distance / self.cfg.grid_size
            features[env_id, 1] = bearing_body / math.pi
            features[env_id, 2] = min(float(cost) / self.cfg.grid_size, 1.0)
            features[env_id, 3] = 1.0 / (1.0 + features[env_id, 2])
            features[env_id, 4] = self._stuck_counter[env_id].float() / max(1, self.cfg.stuck_steps)
        return features

    def _return_home_mask(self):
        return self._steps.float() >= float(self.cfg.max_steps) * self.cfg.return_home_fraction

    def _centers(self):
        return self._pose.round().long().clamp(0, self.cfg.grid_size - 1)

    def _ids(self, env_ids):
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(self.device).long()
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
