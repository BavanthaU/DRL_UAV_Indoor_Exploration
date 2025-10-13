from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


@dataclass
class ClassicalPlannerConfig:
    forward_distance_scale: float = 5.0  # meters needed to reach full forward command
    heading_gain: float = 1.5  # radians corresponding to full yaw command
    turn_slowdown_threshold: float = 0.7  # radians; slow forward motion when turning sharply
    turn_slowdown_factor: float = 0.3
    stop_distance: float = 0.3  # meters
    avoidance_angle: float = 0.75  # radians (~43 deg)
    avoidance_turn_boost: float = 0.8
    lookahead_cells: int = 4
    unknown_penalty: bool = True


class ClassicalLocalPlanner:
    """Lightweight heuristic local planner that generates body-frame velocity commands.

    The planner steers the drone toward the active subgoal selected by the mapping module while
    applying simple collision avoidance based on the occupancy grid in front of the robot.
    """

    def __init__(self, env, cfg: ClassicalPlannerConfig | None = None):
        self.env = getattr(env, "unwrapped", env)
        if not hasattr(self.env, "env_map") or self.env.env_map is None:
            raise RuntimeError("Environment does not expose an env_map with subgoal information.")
        self.env_map = self.env.env_map
        self.device = getattr(self.env, "device", torch.device("cpu"))
        self.cfg = cfg or ClassicalPlannerConfig()
        self.num_envs = self.env.scene.num_envs

        grid_size = float(self.env_map._grid_size)
        distances = torch.arange(1, self.cfg.lookahead_cells + 1, device=self.device, dtype=torch.float32)
        self.lookahead_world = distances * grid_size
        self.grid_size = grid_size
        self.grid_shape = (
            int(self.env_map._grid_num[0]),
            int(self.env_map._grid_num[1]),
        )

    def compute_actions(self) -> torch.Tensor:
        """Return normalized actions in [-1, 1] (forward velocity, yaw rate)."""
        env_map = self.env_map
        root_state = self.env.scene["robot"].data.root_state_w
        pos_xy = root_state[:, :2]
        yaw = euler_xyz_from_quat(root_state[:, 3:7])[2]

        subgoal = env_map.current_subgoal_world[:, :2]
        delta = subgoal - pos_xy
        dist = torch.linalg.norm(delta, dim=1)
        desired_heading = torch.atan2(delta[:, 1], delta[:, 0])
        heading_error = wrap_to_pi(desired_heading - yaw)

        forward_cmd = (dist / self.cfg.forward_distance_scale).clamp(0.0, 1.0)
        yaw_cmd = (heading_error / self.cfg.heading_gain).clamp(-1.0, 1.0)

        large_turn = torch.abs(heading_error) > self.cfg.turn_slowdown_threshold
        forward_cmd = torch.where(large_turn, forward_cmd * self.cfg.turn_slowdown_factor, forward_cmd)

        active = env_map.subgoal_active & (dist > self.cfg.stop_distance)
        forward_cmd = torch.where(active, forward_cmd, torch.zeros_like(forward_cmd))
        yaw_cmd = torch.where(active, yaw_cmd, torch.zeros_like(yaw_cmd))

        blocked_center = self._direction_blocked(pos_xy, yaw)
        blocked_left = self._direction_blocked(pos_xy, yaw + self.cfg.avoidance_angle)
        blocked_right = self._direction_blocked(pos_xy, yaw - self.cfg.avoidance_angle)

        if blocked_center.any():
            forward_cmd = torch.where(blocked_center, torch.zeros_like(forward_cmd), forward_cmd)
            turn_boost = torch.zeros_like(yaw_cmd)
            prefer_left = blocked_center & (~blocked_left)
            prefer_right = blocked_center & blocked_left & (~blocked_right)
            turn_boost = torch.where(prefer_left, torch.ones_like(turn_boost), turn_boost)
            turn_boost = torch.where(prefer_right, -torch.ones_like(turn_boost), turn_boost)
            default_turn = blocked_center & blocked_left & blocked_right
            turn_boost = torch.where(default_turn, torch.sign(yaw_cmd).clamp(min=-1.0, max=1.0), turn_boost)
            yaw_cmd = yaw_cmd + turn_boost * self.cfg.avoidance_turn_boost

        actions = torch.stack(
            (forward_cmd.clamp(-1.0, 1.0), yaw_cmd.clamp(-1.0, 1.0)),
            dim=1,
        )
        return actions

    def _direction_blocked(self, pos_xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        """Return boolean mask indicating whether the path along heading is occupied."""
        if self.lookahead_world.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        dir_vec = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        points = pos_xy.unsqueeze(1) + dir_vec.unsqueeze(1) * self.lookahead_world.view(1, -1, 1)

        grid_orig = self.env_map._grid_orig_tensor.to(self.device)
        grid_coords = torch.floor((points - grid_orig) / self.grid_size).long()

        grid_coords[..., 0] = grid_coords[..., 0].clamp(0, self.grid_shape[0] - 1)
        grid_coords[..., 1] = grid_coords[..., 1].clamp(0, self.grid_shape[1] - 1)

        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, grid_coords.shape[1])
        values = self.env_map._environment_map[
            env_indices,
            grid_coords[..., 0],
            grid_coords[..., 1],
        ]
        obstacle = values == 2
        if self.cfg.unknown_penalty:
            unknown = values == 0
            return obstacle.any(dim=1) | unknown.all(dim=1)
        return obstacle.any(dim=1)
