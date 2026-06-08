from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from exploration_stack.depth_hierarchical.curiosity.temporal_depth_rnd import (
    TemporalDepthEncoder,
    TemporalDepthRNDConfig,
)


@dataclass
class LocalDepthActorCriticConfig:
    history_steps: int = 8
    depth_dim: int = 64
    action_dim: int = 3
    scalar_dim: int = 8
    hidden_dim: int = 128
    feature_dim: int = 256
    max_vx_mps: float = 1.0
    max_vy_mps: float = 0.5
    max_yaw_rate_radps: float = 1.0
    target_altitude_m: float = 1.2
    safe_radius_m: float = 0.45
    danger_radius_m: float = 0.30
    log_std_init: float = -0.5


class LocalPolicyOutput(NamedTuple):
    action: "torch.Tensor"
    log_prob: "torch.Tensor"
    entropy: "torch.Tensor"
    value: "torch.Tensor"
    mean: "torch.Tensor"


class LocalDepthActorCritic(nn.Module if nn is not None else object):
    def __init__(self, cfg: LocalDepthActorCriticConfig | None = None):
        if nn is None:
            raise RuntimeError("LocalDepthActorCritic requires PyTorch.")
        super().__init__()
        self.cfg = cfg or LocalDepthActorCriticConfig()
        encoder_cfg = TemporalDepthRNDConfig(
            depth_dim=self.cfg.depth_dim,
            action_dim=self.cfg.action_dim,
            scalar_dim=self.cfg.scalar_dim,
            hidden_dim=self.cfg.hidden_dim,
            feature_dim=self.cfg.feature_dim,
            use_temporal_conv=True,
            use_gru=False,
        )
        self.encoder = TemporalDepthEncoder(encoder_cfg)
        self.actor = nn.Sequential(nn.LayerNorm(self.cfg.feature_dim), nn.Linear(self.cfg.feature_dim, self.cfg.action_dim))
        self.critic = nn.Sequential(nn.LayerNorm(self.cfg.feature_dim), nn.Linear(self.cfg.feature_dim, 1))
        self.log_std = nn.Parameter(torch.full((self.cfg.action_dim,), float(self.cfg.log_std_init)))
        self.register_buffer(
            "action_scale",
            torch.tensor([self.cfg.max_vx_mps, self.cfg.max_vy_mps, self.cfg.max_yaw_rate_radps], dtype=torch.float32),
        )

    def forward(self, obs_seq, action=None):
        z = self.encoder(obs_seq["depth_rays"], obs_seq["previous_action"], obs_seq["depth_scalars"])
        mean = torch.tanh(self.actor(z)) * self.action_scale
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        raw_action = action if action is not None else dist.rsample()
        bounded_action = torch.max(torch.min(raw_action, self.action_scale), -self.action_scale)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(z).squeeze(-1)
        return LocalPolicyOutput(bounded_action, log_prob, entropy, value, mean)

    def act(self, obs_seq):
        out = self.forward(obs_seq)
        return out.action, {"log_prob": out.log_prob, "value": out.value, "entropy": out.entropy}

    def evaluate_actions(self, obs_seq, actions):
        return self.forward(obs_seq, action=actions)
