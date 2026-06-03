from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class OptionManagerConfig:
    option_interval_steps: int = 12
    max_option_age_steps: int = 20


class OptionManager:
    def __init__(self, num_envs: int, cfg: OptionManagerConfig | None = None, device="cpu"):
        if torch is None:
            raise RuntimeError("OptionManager requires PyTorch.")
        self.cfg = cfg or OptionManagerConfig()
        self.device = torch.device(device)
        self.option = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.candidate = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.option.shape[0], device=self.device) if env_ids is None else env_ids.to(self.device)
        self.option[ids] = 0
        self.candidate[ids] = 0
        self.age[ids] = self.cfg.option_interval_steps

    def should_resample(self, termination_prob):
        periodic = self.age >= self.cfg.option_interval_steps
        too_old = self.age >= self.cfg.max_option_age_steps
        terminate = termination_prob > 0.5
        return periodic | too_old | terminate

    def update(self, option, candidate, resample_mask):
        self.option = torch.where(resample_mask, option, self.option)
        self.candidate = torch.where(resample_mask, candidate, self.candidate)
        self.age = torch.where(resample_mask, torch.zeros_like(self.age), self.age + 1)
