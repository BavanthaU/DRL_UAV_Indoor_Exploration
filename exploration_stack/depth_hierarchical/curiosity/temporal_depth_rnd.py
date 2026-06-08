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

from .running_stats import RunningMeanStd


@dataclass
class TemporalDepthRNDConfig:
    depth_dim: int = 64
    action_dim: int = 3
    scalar_dim: int = 8
    hidden_dim: int = 128
    feature_dim: int = 256
    use_temporal_conv: bool = True
    use_gru: bool = False
    reward_clip: float = 1.0
    rnd_lr: float = 1.0e-4
    normalize_obs: bool = True
    normalize_reward: bool = True
    zero_on_collision: bool = True
    zero_if_min_depth_below: float = 0.30


class TemporalDepthEncoder(nn.Module if nn is not None else object):
    def __init__(self, cfg: TemporalDepthRNDConfig):
        if nn is None:
            raise RuntimeError("TemporalDepthEncoder requires PyTorch.")
        super().__init__()
        self.cfg = cfg
        part = cfg.hidden_dim // 3
        rem = cfg.hidden_dim - 2 * part
        self.depth_proj = nn.Sequential(nn.Linear(cfg.depth_dim, part), nn.LayerNorm(part), nn.SiLU())
        self.action_proj = nn.Sequential(nn.Linear(cfg.action_dim, part), nn.LayerNorm(part), nn.SiLU())
        self.scalar_proj = nn.Sequential(nn.Linear(cfg.scalar_dim, rem), nn.LayerNorm(rem), nn.SiLU())
        if cfg.use_gru:
            self.temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
            self.temporal_conv = None
        elif cfg.use_temporal_conv:
            self.temporal = None
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.SiLU(),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.SiLU(),
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=4, dilation=4),
                nn.SiLU(),
            )
        else:
            self.temporal = None
            self.temporal_conv = nn.Sequential(nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1), nn.SiLU())
        self.attn = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(), nn.Linear(cfg.hidden_dim, 1))
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.feature_dim))

    def forward(self, depth_rays, previous_action, depth_scalars):
        depth_rays = torch.nan_to_num(depth_rays.float(), nan=0.0, posinf=0.0, neginf=0.0)
        previous_action = torch.nan_to_num(previous_action.float(), nan=0.0, posinf=0.0, neginf=0.0)
        depth_scalars = torch.nan_to_num(depth_scalars.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat(
            [self.depth_proj(depth_rays), self.action_proj(previous_action), self.scalar_proj(depth_scalars)],
            dim=-1,
        )
        if self.temporal is not None:
            x, _ = self.temporal(x)
        else:
            x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        pooled = (x * weights).sum(dim=1)
        return self.out(pooled)


class TemporalDepthRND(nn.Module if nn is not None else object):
    def __init__(self, cfg: TemporalDepthRNDConfig | None = None, device="cpu"):
        if nn is None:
            raise RuntimeError("TemporalDepthRND requires PyTorch.")
        super().__init__()
        self.cfg = cfg or TemporalDepthRNDConfig()
        self.target_encoder = TemporalDepthEncoder(self.cfg)
        self.predictor_encoder = TemporalDepthEncoder(self.cfg)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.target_encoder.eval()
        self.reward_rms = RunningMeanStd().to(device)
        self.optimizer = torch.optim.Adam(self.predictor_encoder.parameters(), lr=self.cfg.rnd_lr)
        self.to(device)

    def _encode(self, obs_seq, *, train_predictor: bool):
        depth_rays = obs_seq["depth_rays"]
        previous_action = obs_seq["previous_action"]
        depth_scalars = obs_seq["depth_scalars"]
        if self.cfg.normalize_obs:
            depth_rays = torch.clamp(depth_rays, 0.0, 20.0) / 20.0
            previous_action = torch.clamp(previous_action, -2.0, 2.0) / 2.0
            depth_scalars = torch.nan_to_num(depth_scalars.float(), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            target_z = self.target_encoder(depth_rays, previous_action, depth_scalars)
        if train_predictor:
            pred_z = self.predictor_encoder(depth_rays, previous_action, depth_scalars)
        else:
            with torch.set_grad_enabled(self.predictor_encoder.training):
                pred_z = self.predictor_encoder(depth_rays, previous_action, depth_scalars)
        return pred_z, target_z

    def forward(self, obs_seq, *, collision=None, min_depth=None, update_stats: bool = True):
        pred_z, target_z = self._encode(obs_seq, train_predictor=False)
        raw_error = F.mse_loss(pred_z, target_z, reduction="none").mean(dim=-1)
        if update_stats and self.cfg.normalize_reward:
            self.reward_rms.update(raw_error.detach())
        reward = self.reward_rms.normalize(raw_error) if self.cfg.normalize_reward else raw_error
        reward = reward.clamp(0.0, self.cfg.reward_clip)
        if self.cfg.zero_on_collision and collision is not None:
            reward = torch.where(collision.bool(), torch.zeros_like(reward), reward)
        if min_depth is not None:
            reward = torch.where(min_depth < self.cfg.zero_if_min_depth_below, torch.zeros_like(reward), reward)
        return reward, raw_error.detach()

    def update_predictor(self, obs_seq):
        pred_z, target_z = self._encode(obs_seq, train_predictor=True)
        loss = F.mse_loss(pred_z, target_z)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())
