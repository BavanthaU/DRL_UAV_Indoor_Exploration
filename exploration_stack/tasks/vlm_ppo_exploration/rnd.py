from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .reward_normalizer import RunningMeanStd


@dataclass
class RNDConfig:
    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 64
    learning_rate: float = 1.0e-4
    reward_clip: float = 5.0


class RNDIntrinsicReward(nn.Module if nn is not None else object):
    def __init__(self, cfg: RNDConfig, device="cpu"):
        if nn is None:
            raise RuntimeError("RNDIntrinsicReward requires PyTorch.")
        super().__init__()
        self.cfg = cfg
        self.target = self._mlp(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
        self.predictor = self._mlp(cfg.input_dim, cfg.hidden_dim, cfg.output_dim)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.error_rms = RunningMeanStd().to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=cfg.learning_rate)
        self.to(device)

    @staticmethod
    def _mlp(input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.detach().float()
        with torch.no_grad():
            target = self.target(x)
        prediction = self.predictor(x)
        raw_error = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        self.error_rms.update(raw_error.detach())
        normalized = self.error_rms.normalize(raw_error).clamp(0.0, self.cfg.reward_clip)
        return normalized, raw_error

    def update_predictor(self, x):
        x = x.detach().float()
        with torch.no_grad():
            target = self.target(x)
        prediction = self.predictor(x)
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())
