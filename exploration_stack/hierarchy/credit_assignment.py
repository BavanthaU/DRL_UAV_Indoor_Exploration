from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class OptionCreditConfig:
    collision_penalty: float = 10.0
    stuck_penalty: float = 0.5
    revisit_penalty: float = 0.2
    path_inefficiency_penalty: float = 0.1


class OptionCreditAccumulator:
    def __init__(self, num_envs: int, cfg: OptionCreditConfig | None = None, device="cpu"):
        if torch is None:
            raise RuntimeError("OptionCreditAccumulator requires PyTorch.")
        self.cfg = cfg or OptionCreditConfig()
        self.device = torch.device(device)
        self.return_sum = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.duration = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.return_sum.shape[0], device=self.device) if env_ids is None else env_ids.to(self.device)
        self.return_sum[ids] = 0.0
        self.duration[ids] = 0

    def update(self, *, map_progress, new_cell_reward, collision, stuck, revisit_ratio=None, path_inefficiency=None):
        if revisit_ratio is None:
            revisit_ratio = torch.zeros_like(map_progress)
        if path_inefficiency is None:
            path_inefficiency = torch.zeros_like(map_progress)
        reward = (
            map_progress
            + new_cell_reward
            - self.cfg.collision_penalty * collision.float()
            - self.cfg.stuck_penalty * stuck.float()
            - self.cfg.revisit_penalty * revisit_ratio
            - self.cfg.path_inefficiency_penalty * path_inefficiency
        )
        self.return_sum += reward
        self.duration += 1
        return reward

    def pop_completed(self, completed):
        rewards = self.return_sum.clone()
        durations = self.duration.clone()
        self.return_sum[completed] = 0.0
        self.duration[completed] = 0
        return rewards, durations
