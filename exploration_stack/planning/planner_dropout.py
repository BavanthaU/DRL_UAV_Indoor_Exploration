from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class PlannerDropoutConfig:
    planner_dropout_prob: float = 0.2
    astar_feature_dropout_prob: float = 0.2
    frontier_candidate_dropout_prob: float = 0.1


class PlannerFeatureDropout:
    """Drop planner-derived features/proposals so learned policy cannot over-rely on them."""

    def __init__(self, cfg: PlannerDropoutConfig | None = None):
        if torch is None:
            raise RuntimeError("PlannerFeatureDropout requires PyTorch.")
        self.cfg = cfg or PlannerDropoutConfig()

    def __call__(self, candidate_features, candidate_mask, *, training: bool = True):
        if not training:
            return torch.nan_to_num(candidate_features, nan=0.0, posinf=0.0, neginf=0.0), candidate_mask
        features = torch.nan_to_num(candidate_features.clone(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = candidate_mask.clone()
        original_mask = mask.clone()
        if self.cfg.planner_dropout_prob > 0.0:
            planner_mask = torch.rand(features.shape[:2], device=features.device) < self.cfg.planner_dropout_prob
            features[..., 8:12] = features[..., 8:12].masked_fill(planner_mask.unsqueeze(-1), 0.0)
        if self.cfg.astar_feature_dropout_prob > 0.0:
            astar_mask = torch.rand(features.shape[:2], device=features.device) < self.cfg.astar_feature_dropout_prob
            features[..., 8] = features[..., 8].masked_fill(astar_mask, 0.0)
        if self.cfg.frontier_candidate_dropout_prob > 0.0:
            candidate_drop = torch.rand(mask.shape, device=mask.device) < self.cfg.frontier_candidate_dropout_prob
            candidate_drop[:, 0] = False
            mask &= ~candidate_drop
            empty = ~mask.any(dim=1)
            if empty.any():
                restore_idx = original_mask.float().argmax(dim=1)
                rows = torch.nonzero(empty, as_tuple=False).flatten()
                mask[rows, restore_idx[rows]] = original_mask[rows, restore_idx[rows]]
                still_empty = ~mask.any(dim=1)
                if still_empty.any():
                    mask[still_empty, 0] = True
        return features, mask
