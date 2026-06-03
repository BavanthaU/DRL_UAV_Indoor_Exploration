from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class RunningMeanStd:
    epsilon: float = 1e-4
    shape: tuple[int, ...] = ()

    def __post_init__(self):
        if torch is None:
            raise RuntimeError("RunningMeanStd requires PyTorch.")
        self.mean = torch.zeros(self.shape, dtype=torch.float32)
        self.var = torch.ones(self.shape, dtype=torch.float32)
        self.count = torch.tensor(self.epsilon, dtype=torch.float32)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.count = self.count.to(device)
        return self

    def update(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=torch.float32)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device).clamp_min(1e-8))

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


class RewardNormalizer:
    def __init__(self, clip: float = 5.0):
        self.rms = RunningMeanStd()
        self.clip = clip

    def to(self, device):
        self.rms.to(device)
        return self

    def __call__(self, reward):
        reward = torch.nan_to_num(reward.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.rms.update(reward.detach().reshape(-1))
        return torch.clamp(self.rms.normalize(reward), -self.clip, self.clip)
