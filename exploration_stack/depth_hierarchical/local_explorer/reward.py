from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class LocalRewardConfig:
    beta_depth_rnd: float = 1.0
    w_collision: float = 8.0
    w_near_obstacle: float = 0.5
    w_altitude: float = 0.2
    w_action_smoothness: float = 0.02
    w_time: float = 0.001


def compute_local_reward(
    depth_rnd_reward,
    *,
    collision,
    near_obstacle_penalty,
    altitude_error,
    action_smoothness,
    cfg: LocalRewardConfig | None = None,
    new_cells_metric_only=None,
):
    """Depth-local reward.

    ``new_cells_metric_only`` is accepted to make call sites explicit, but it is
    intentionally unused: map gain is not part of the local reward.
    """

    if torch is None:
        raise RuntimeError("compute_local_reward requires PyTorch.")
    cfg = cfg or LocalRewardConfig()
    reward = cfg.beta_depth_rnd * depth_rnd_reward.float()
    reward = reward - cfg.w_collision * collision.float()
    reward = reward - cfg.w_near_obstacle * near_obstacle_penalty.float()
    reward = reward - cfg.w_altitude * altitude_error.float()
    reward = reward - cfg.w_action_smoothness * action_smoothness.float()
    reward = reward - cfg.w_time
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
