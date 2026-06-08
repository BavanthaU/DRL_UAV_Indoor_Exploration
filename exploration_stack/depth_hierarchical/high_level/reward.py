from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class HighLevelRewardConfig:
    beta_map_rnd: float = 0.5
    w_gain: float = 2.0
    w_missing: float = 0.5
    w_region: float = 1.0
    w_path: float = 0.5
    w_unreachable: float = 2.0
    w_no_gain: float = 1.0
    w_collision: float = 5.0
    w_revisit: float = 0.5
    expected_candidate_gain: float = 500.0


def compute_high_level_reward(
    *,
    map_rnd_reward,
    new_free_cells,
    unknown_to_known_cells,
    missing_patch_filled,
    new_connected_region_entered,
    astar_path_length,
    unreachable,
    no_gain_after_reach,
    collision_during_transit_or_local,
    revisit_ratio_after_candidate,
    cfg: HighLevelRewardConfig | None = None,
):
    if torch is None:
        raise RuntimeError("compute_high_level_reward requires PyTorch.")
    cfg = cfg or HighLevelRewardConfig()
    gain = (new_free_cells.float() + 0.5 * unknown_to_known_cells.float()) / float(cfg.expected_candidate_gain)
    gain = gain.clamp(0.0, 1.0)
    normalized_path = astar_path_length.float() / float(cfg.expected_candidate_gain)
    reward = cfg.beta_map_rnd * map_rnd_reward.float()
    reward = reward + cfg.w_gain * gain
    reward = reward + cfg.w_missing * missing_patch_filled.float()
    reward = reward + cfg.w_region * new_connected_region_entered.float()
    reward = reward - cfg.w_path * normalized_path
    reward = reward - cfg.w_unreachable * unreachable.float()
    reward = reward - cfg.w_no_gain * no_gain_after_reach.float()
    reward = reward - cfg.w_collision * collision_during_transit_or_local.float()
    reward = reward - cfg.w_revisit * revisit_ratio_after_candidate.float()
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), gain
