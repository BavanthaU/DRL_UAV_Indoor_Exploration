from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.depth_hierarchical.map_memory import FREE, OCCUPIED, UNKNOWN, MapMemory, MapMemoryConfig


@dataclass
class DepthHierarchicalMockEnvConfig:
    num_envs: int = 2
    depth_dim: int = 64
    history_steps: int = 8
    scalar_dim: int = 8
    action_dim: int = 3
    max_steps: int = 64
    map_size: int = 32
    device: str = "cpu"
    seed: int = 7


class DepthHierarchicalMockEnv:
    """Fast deterministic stand-in for Isaac smoke tests and one-update runs."""

    def __init__(self, cfg: DepthHierarchicalMockEnvConfig | None = None, **kwargs):
        if torch is None:
            raise RuntimeError("DepthHierarchicalMockEnv requires PyTorch.")
        self.cfg = cfg or DepthHierarchicalMockEnvConfig(**kwargs)
        self.num_envs = self.cfg.num_envs
        self.device = torch.device(self.cfg.device)
        self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pose = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.cfg.history_steps, self.cfg.action_dim, device=self.device)
        self.map_memory = MapMemory(MapMemoryConfig(height=self.cfg.map_size, width=self.cfg.map_size))

    def reset(self):
        self.step_count.zero_()
        self.pose.zero_()
        center = self.cfg.map_size // 2
        self.map_memory = MapMemory(MapMemoryConfig(height=self.cfg.map_size, width=self.cfg.map_size))
        self.map_memory.mark_observed(
            free_cells=[(r, c) for r in range(center - 2, center + 3) for c in range(center - 2, center + 3)],
            occupied_cells=[(center - 3, center), (center + 3, center)],
            agent_cell=(center, center),
        )
        return self._obs()

    def step(self, action):
        action = torch.nan_to_num(action.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_action = torch.roll(self.prev_action, shifts=-1, dims=1)
        self.prev_action[:, -1] = action
        self.pose[:, 0] += action[:, 0] * 0.1
        self.pose[:, 1] += action[:, 1] * 0.1
        self.pose[:, 2] += action[:, 2] * 0.1
        self.step_count += 1
        center = self.cfg.map_size // 2
        free = []
        for env_id in range(self.num_envs):
            row = int(center + round(float(self.pose[env_id, 1].cpu())))
            col = int(center + round(float(self.pose[env_id, 0].cpu())))
            for radius in range(1, 4):
                free.append((row + radius, col))
                free.append((row, col + radius))
        new_cells = self.map_memory.mark_observed(free_cells=free, agent_cell=(center, center))
        depth_min = self._depth_min()
        collision = depth_min < 0.18
        reward = torch.full((self.num_envs,), float(new_cells) / 50.0, device=self.device) - collision.float()
        done = self.step_count >= self.cfg.max_steps
        info = {
            "new_cells": torch.full((self.num_envs,), float(new_cells), device=self.device),
            "collision": collision,
            "coverage_percent": torch.full((self.num_envs,), self.coverage_percent(), device=self.device),
        }
        return self._obs(), reward, done, info

    def coverage_percent(self):
        grid = self.map_memory.grid
        return float(((grid != UNKNOWN).sum() / grid.size) * 100.0)

    def _depth_min(self):
        return self._obs()["depth_scalars"][:, -1, 0]

    def _obs(self):
        batch = self.num_envs
        t = self.cfg.history_steps
        rays = torch.linspace(0.2, 4.0, self.cfg.depth_dim, device=self.device).repeat(batch, t, 1)
        phase = self.step_count.float().view(batch, 1, 1) * 0.05
        angles = torch.linspace(-1.2, 1.2, self.cfg.depth_dim, device=self.device).view(1, 1, -1)
        rays = rays + 0.4 * torch.sin(angles + phase)
        rays = rays.clamp_min(0.05)
        scalars = torch.zeros(batch, t, self.cfg.scalar_dim, device=self.device)
        scalars[..., 0] = rays.min(dim=-1).values
        scalars[..., 1] = rays.mean(dim=-1)
        scalars[..., 2] = rays.std(dim=-1)
        split = self.cfg.depth_dim // 3
        scalars[..., 3] = rays[..., :split].mean(dim=-1)
        scalars[..., 4] = rays[..., split : 2 * split].mean(dim=-1)
        scalars[..., 5] = rays[..., 2 * split :].mean(dim=-1)
        scalars[..., 6] = 0.05
        scalars[..., 7] = 0.0
        return {"depth_rays": rays, "previous_action": self.prev_action.clone(), "depth_scalars": scalars}
