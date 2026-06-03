from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class NewCellCuriosityConfig:
    max_new_cells_per_step: int = 64
    free_value: int = 1
    include_unknown_to_free: bool = True


class NewCellCountCuriosity:
    """Per-environment visited-free-cell memory and normalized new-cell reward."""

    def __init__(self, num_envs: int, grid_shape: tuple[int, int], cfg: NewCellCuriosityConfig | None = None, device="cpu"):
        if torch is None:
            raise RuntimeError("NewCellCountCuriosity requires PyTorch.")
        self.cfg = cfg or NewCellCuriosityConfig()
        self.num_envs = num_envs
        self.grid_shape = grid_shape
        self.device = torch.device(device)
        self.visited = torch.zeros((num_envs, *grid_shape), dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.visited.zero_()
        else:
            self.visited[env_ids] = False

    def update(self, occupancy_grid):
        if occupancy_grid.ndim != 3:
            raise ValueError("occupancy_grid must be [B,H,W]")
        free = occupancy_grid.to(self.device) == self.cfg.free_value
        newly_observed = free & ~self.visited
        new_counts = newly_observed.flatten(start_dim=1).sum(dim=1).float()
        self.visited |= free
        denom = float(max(1, self.cfg.max_new_cells_per_step)) ** 0.5
        reward = torch.sqrt(new_counts.clamp_min(0.0)) / denom
        return reward.clamp(0.0, 1.0), new_counts


class SemanticNovelty:
    """Count-based novelty over prompt-similarity bins and local map sector."""

    def __init__(self, num_envs: int, bins: int = 8):
        self.num_envs = num_envs
        self.bins = bins
        self.counts = [dict() for _ in range(num_envs)]

    def reset(self, env_ids=None):
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for env_id in ids:
            self.counts[env_id].clear()

    def update(self, prompt_similarity, sector_ids=None):
        probs = torch.softmax(prompt_similarity.detach(), dim=-1)
        top = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        conf_bin = torch.clamp((confidence * self.bins).long(), 0, self.bins - 1)
        if sector_ids is None:
            sector_ids = torch.zeros_like(top)
        rewards = []
        for env_id in range(prompt_similarity.shape[0]):
            key = (int(top[env_id].item()), int(conf_bin[env_id].item()), int(sector_ids[env_id].item()))
            count = self.counts[env_id].get(key, 0) + 1
            self.counts[env_id][key] = count
            rewards.append(1.0 / (count**0.5))
        return torch.tensor(rewards, dtype=torch.float32, device=prompt_similarity.device)

