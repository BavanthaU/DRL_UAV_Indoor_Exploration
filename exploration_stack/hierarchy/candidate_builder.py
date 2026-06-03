from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.planning.astar_features import approximate_path_features
from exploration_stack.planning.frontier_candidates import frontier_centroids_from_mask

from .option_types import CandidateType, NUM_CANDIDATE_TYPES


@dataclass
class CandidateBuilderConfig:
    max_candidates: int = 16
    feature_dim: int = 18
    planner_feature_mode: str = "features_only"
    use_privileged_map_for_training: bool = False


class CandidateBuilder:
    """Build padded candidate sets for the learned option policy.

    The candidates are proposals/features only. They are not commands and do
    not directly drive the UAV.
    """

    def __init__(self, cfg: CandidateBuilderConfig | None = None):
        if torch is None:
            raise RuntimeError("CandidateBuilder requires PyTorch.")
        self.cfg = cfg or CandidateBuilderConfig()
        if self.cfg.max_candidates < 4:
            raise ValueError("CandidateBuilderConfig.max_candidates must be at least 4.")
        if self.cfg.feature_dim < 14:
            raise ValueError("CandidateBuilderConfig.feature_dim must be at least 14.")
        if self.cfg.planner_feature_mode not in {"none", "features_only", "proposal_only", "oracle_baseline"}:
            raise ValueError(
                "planner_feature_mode must be one of none, features_only, proposal_only, or oracle_baseline."
            )

    def build(self, obs: dict[str, "torch.Tensor"]):
        if self.cfg.planner_feature_mode == "none":
            return self._fallback_candidates(obs)
        frontier_mask = obs.get("frontier_mask")
        if frontier_mask is None:
            return self._fallback_candidates(obs)
        frontier_mask = frontier_mask.float()
        batch, height, width = frontier_mask.shape
        features = torch.zeros(batch, self.cfg.max_candidates, self.cfg.feature_dim, device=frontier_mask.device)
        mask = torch.zeros(batch, self.cfg.max_candidates, dtype=torch.bool, device=frontier_mask.device)
        centroids, frontier_valid = frontier_centroids_from_mask(frontier_mask, max_candidates=max(1, self.cfg.max_candidates - 3))
        center = torch.tensor([height // 2, width // 2], dtype=torch.float32, device=frontier_mask.device)
        frontier_count = centroids.shape[1]
        if frontier_count:
            rel = (centroids - center) / float(max(height, width))
            distance = torch.linalg.norm(rel, dim=-1)
            bearing = torch.atan2(rel[..., 0], rel[..., 1]) / torch.pi
            path = approximate_path_features(rel, max(height, width))
            features[:, :frontier_count, :NUM_CANDIDATE_TYPES] = self._type_one_hot(
                CandidateType.FRONTIER,
                (batch, frontier_count),
                frontier_mask.device,
            )
            features[:, :frontier_count, 8] = path[..., 0]
            features[:, :frontier_count, 9] = path[..., 1]
            features[:, :frontier_count, 10] = path[..., 2]
            features[:, :frontier_count, 11] = distance
            features[:, :frontier_count, 12] = bearing
            features[:, :frontier_count, 13] = 1.0 - distance.clamp(0.0, 1.0)
            if self.cfg.planner_feature_mode == "proposal_only":
                features[:, :frontier_count, 8:14] = 0.0
            mask[:, :frontier_count] = frontier_valid
        self._add_scan_recovery_stop(features, mask, start=frontier_count)
        if not mask.any(dim=1).all():
            missing = ~mask.any(dim=1)
            features[missing, 0, :NUM_CANDIDATE_TYPES] = self._type_one_hot(
                CandidateType.ROTATE_SCAN,
                (int(missing.sum().item()), 1),
                frontier_mask.device,
            ).squeeze(1)
            mask[missing, 0] = True
        return features, mask

    def _fallback_candidates(self, obs):
        reference = next(iter(obs.values()))
        batch = reference.shape[0]
        features = torch.zeros(batch, self.cfg.max_candidates, self.cfg.feature_dim, device=reference.device)
        mask = torch.zeros(batch, self.cfg.max_candidates, dtype=torch.bool, device=reference.device)
        self._add_scan_recovery_stop(features, mask, start=0)
        return features, mask

    def _add_scan_recovery_stop(self, features, mask, *, start: int):
        batch = features.shape[0]
        specials = [CandidateType.ROTATE_SCAN, CandidateType.AVOID_AND_RECOVER, CandidateType.STOP]
        for offset, candidate_type in enumerate(specials):
            idx = min(start + offset, self.cfg.max_candidates - 1)
            features[:, idx, :NUM_CANDIDATE_TYPES] = self._type_one_hot(candidate_type, (batch, 1), features.device).squeeze(1)
            features[:, idx, 9] = 1.0
            features[:, idx, 10] = 1.0
            mask[:, idx] = True

    @staticmethod
    def _type_one_hot(candidate_type: CandidateType, shape: tuple[int, int], device):
        out = torch.zeros(*shape, NUM_CANDIDATE_TYPES, device=device)
        out[..., int(candidate_type)] = 1.0
        return out
