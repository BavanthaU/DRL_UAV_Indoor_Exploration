from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class HighLevelCandidateSelectorPPOConfig:
    candidate_feature_dim: int = 24
    max_candidates: int = 64
    map_channels: int = 4
    global_stats_dim: int = 8
    hidden_dim: int = 128
    heads: int = 4
    transformer_layers: int = 2


class SelectorOutput(NamedTuple):
    selected_index: "torch.Tensor"
    log_prob: "torch.Tensor"
    entropy: "torch.Tensor"
    value: "torch.Tensor"
    logits: "torch.Tensor"
    probabilities: "torch.Tensor"


class HighLevelCandidateSelectorPPO(nn.Module if nn is not None else object):
    """Masked categorical policy over the complete padded candidate set."""

    def __init__(self, cfg: HighLevelCandidateSelectorPPOConfig | None = None):
        if nn is None:
            raise RuntimeError("HighLevelCandidateSelectorPPO requires PyTorch.")
        super().__init__()
        self.cfg = cfg or HighLevelCandidateSelectorPPOConfig()
        self.candidate_proj = nn.Sequential(
            nn.Linear(self.cfg.candidate_feature_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_dim,
            nhead=self.cfg.heads,
            dim_feedforward=self.cfg.hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.transformer_layers)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.cfg.map_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, self.cfg.hidden_dim),
            nn.SiLU(),
        )
        self.global_proj = nn.Sequential(nn.Linear(self.cfg.global_stats_dim, self.cfg.hidden_dim), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(self.cfg.hidden_dim * 3, self.cfg.hidden_dim), nn.SiLU())
        self.policy_head = nn.Linear(self.cfg.hidden_dim * 2, 1)
        self.value_head = nn.Sequential(nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim), nn.SiLU(), nn.Linear(self.cfg.hidden_dim, 1))

    def forward(self, candidate_features, candidate_mask, map_crop, global_map_stats, selected_index=None):
        candidate_features = torch.nan_to_num(candidate_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        candidate_mask = candidate_mask.bool().clone()
        empty = ~candidate_mask.any(dim=1)
        if empty.any():
            candidate_mask[empty, 0] = True
        encoded = self.candidate_proj(candidate_features)
        encoded = self.candidate_encoder(encoded, src_key_padding_mask=~candidate_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        masked = encoded.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        pooled = masked.sum(dim=1) / candidate_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        context = self.fusion(torch.cat([pooled, self.map_encoder(map_crop), self.global_proj(global_map_stats.float())], dim=-1))
        logits = self.policy_head(torch.cat([encoded, context.unsqueeze(1).expand_as(encoded)], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1.0e9)
        dist = torch.distributions.Categorical(logits=logits)
        selected = dist.sample() if selected_index is None else selected_index.long()
        return SelectorOutput(
            selected_index=selected,
            log_prob=dist.log_prob(selected),
            entropy=dist.entropy(),
            value=self.value_head(context).squeeze(-1),
            logits=logits,
            probabilities=dist.probs.masked_fill(~candidate_mask, 0.0),
        )

    def evaluate_actions(self, candidate_features, candidate_mask, map_crop, global_map_stats, selected_index):
        return self.forward(candidate_features, candidate_mask, map_crop, global_map_stats, selected_index=selected_index)
