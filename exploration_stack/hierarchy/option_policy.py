from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from .option_types import NUM_OPTIONS


@dataclass
class OptionPolicyConfig:
    context_dim: int = 256
    candidate_dim: int = 128
    hidden_dim: int = 256
    num_options: int = NUM_OPTIONS


class SemanticOptionPolicy(nn.Module if nn is not None else object):
    def __init__(self, cfg: OptionPolicyConfig | None = None):
        if nn is None:
            raise RuntimeError("SemanticOptionPolicy requires PyTorch.")
        super().__init__()
        self.cfg = cfg or OptionPolicyConfig()
        self.context_proj = nn.Sequential(
            nn.Linear(self.cfg.context_dim + self.cfg.candidate_dim, self.cfg.hidden_dim),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.SiLU(),
        )
        self.option_head = nn.Linear(self.cfg.hidden_dim, self.cfg.num_options)
        self.candidate_query = nn.Linear(self.cfg.hidden_dim, self.cfg.candidate_dim)
        self.termination_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.confidence_head = nn.Linear(self.cfg.hidden_dim, 1)

    def forward(self, context, candidate_tokens, candidate_pooled, candidate_mask):
        hidden = self.context_proj(torch.cat([context, candidate_pooled], dim=-1))
        option_logits = self.option_head(hidden)
        query = self.candidate_query(hidden).unsqueeze(1)
        candidate_logits = torch.sum(candidate_tokens * query, dim=-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask.bool(), -1.0e9)
        termination_prob = torch.sigmoid(self.termination_head(hidden)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1)
        return option_logits, candidate_logits, termination_prob, confidence
