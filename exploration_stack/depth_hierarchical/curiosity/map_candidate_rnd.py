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
class MapCandidateRNDConfig:
    map_channels: int = 4
    candidate_feature_dim: int = 24
    global_stats_dim: int = 8
    previous_outcome_dim: int = 6
    hidden_dim: int = 128
    feature_dim: int = 256
    rnd_lr: float = 1.0e-4
    reward_clip: float = 1.0
    normalize_reward: bool = True


class MapCandidateEncoder(nn.Module if nn is not None else object):
    def __init__(self, cfg: MapCandidateRNDConfig):
        if nn is None:
            raise RuntimeError("MapCandidateEncoder requires PyTorch.")
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(cfg.map_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(cfg.candidate_feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(cfg.global_stats_dim + cfg.previous_outcome_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.feature_dim),
        )

    def forward(self, map_crop, candidate_features, global_stats, previous_outcome):
        map_crop = torch.nan_to_num(map_crop.float(), nan=0.0, posinf=0.0, neginf=0.0)
        candidate_features = torch.nan_to_num(candidate_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        global_stats = torch.nan_to_num(global_stats.float(), nan=0.0, posinf=0.0, neginf=0.0)
        previous_outcome = torch.nan_to_num(previous_outcome.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return self.fusion(
            torch.cat(
                [
                    self.map_encoder(map_crop),
                    self.candidate_proj(candidate_features),
                    self.global_proj(torch.cat([global_stats, previous_outcome], dim=-1)),
                ],
                dim=-1,
            )
        )


class MapCandidateRND(nn.Module if nn is not None else object):
    def __init__(self, cfg: MapCandidateRNDConfig | None = None, device="cpu"):
        if nn is None:
            raise RuntimeError("MapCandidateRND requires PyTorch.")
        super().__init__()
        self.cfg = cfg or MapCandidateRNDConfig()
        self.target_encoder = MapCandidateEncoder(self.cfg)
        self.predictor_encoder = MapCandidateEncoder(self.cfg)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.target_encoder.eval()
        self.reward_rms = RunningMeanStd().to(device)
        self.optimizer = torch.optim.Adam(self.predictor_encoder.parameters(), lr=self.cfg.rnd_lr)
        self.to(device)

    def forward(self, map_crop, candidate_features, global_stats, previous_outcome, *, update_stats: bool = True):
        with torch.no_grad():
            target_z = self.target_encoder(map_crop, candidate_features, global_stats, previous_outcome)
        pred_z = self.predictor_encoder(map_crop, candidate_features, global_stats, previous_outcome)
        raw_error = F.mse_loss(pred_z, target_z, reduction="none").mean(dim=-1)
        if update_stats and self.cfg.normalize_reward:
            self.reward_rms.update(raw_error.detach())
        reward = self.reward_rms.normalize(raw_error) if self.cfg.normalize_reward else raw_error
        return reward.clamp(0.0, self.cfg.reward_clip), raw_error.detach()

    def update_predictor(self, map_crop, candidate_features, global_stats, previous_outcome):
        with torch.no_grad():
            target_z = self.target_encoder(map_crop, candidate_features, global_stats, previous_outcome)
        pred_z = self.predictor_encoder(map_crop, candidate_features, global_stats, previous_outcome)
        loss = F.mse_loss(pred_z, target_z)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())
