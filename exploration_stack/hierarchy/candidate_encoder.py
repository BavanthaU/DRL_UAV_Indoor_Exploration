from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class CandidateEncoderConfig:
    feature_dim: int = 18
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 1


class CandidateSetEncoder(nn.Module if nn is not None else object):
    def __init__(self, cfg: CandidateEncoderConfig | None = None):
        if nn is None:
            raise RuntimeError("CandidateSetEncoder requires PyTorch.")
        super().__init__()
        self.cfg = cfg or CandidateEncoderConfig()
        self.input_proj = nn.Sequential(
            nn.Linear(self.cfg.feature_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_dim,
            nhead=self.cfg.heads,
            dim_feedforward=self.cfg.hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.layers)

    def forward(self, candidate_features, candidate_mask):
        candidate_features = torch.nan_to_num(candidate_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        candidate_mask = candidate_mask.bool().clone()
        empty = ~candidate_mask.any(dim=1)
        if empty.any():
            candidate_mask[empty, 0] = True
        encoded = self.input_proj(candidate_features)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        key_padding_mask = ~candidate_mask
        encoded = self.encoder(encoded, src_key_padding_mask=key_padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        masked = encoded.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        denom = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        pooled = masked.sum(dim=1) / denom
        return encoded, pooled
