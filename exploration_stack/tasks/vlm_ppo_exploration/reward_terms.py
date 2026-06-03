from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class RewardWeights:
    w_frontier_closure: float = 5.0
    w_frontier_progress: float = 0.2
    w_map_progress: float = 1.0
    w_return_home_progress: float = 1.0
    w_collision: float = 10.0
    w_near_obstacle: float = 1.0
    w_time: float = 0.002
    w_idle: float = 0.5
    w_oscillation: float = 0.05
    w_action_smoothness: float = 0.02
    w_altitude_error: float = 0.2
    beta_count: float = 1.0
    beta_rnd: float = 0.05
    beta_semantic_novelty: float = 0.02


def compute_extrinsic_reward(
    *,
    map_progress,
    frontier_closed,
    collision,
    esdf_clearance,
    safety_radius,
    dt,
    idle_mask,
    yaw_flip,
    action,
    prev_action,
    altitude,
    target_altitude,
    frontier_distance_delta=None,
    return_home_progress=None,
    weights: RewardWeights | None = None,
):
    weights = weights or RewardWeights()
    if frontier_distance_delta is None:
        frontier_distance_delta = torch.zeros_like(map_progress)
    if return_home_progress is None:
        return_home_progress = torch.zeros_like(map_progress)
    near_obstacle = (float(safety_radius) - esdf_clearance).clamp_min(0.0)
    smoothness = torch.sum(torch.square(action - prev_action), dim=-1)
    altitude_error = torch.abs(altitude - target_altitude)
    terms = {
        "frontier_closure": weights.w_frontier_closure * frontier_closed.float(),
        "frontier_progress": weights.w_frontier_progress * frontier_distance_delta,
        "map_progress": weights.w_map_progress * map_progress,
        "return_home_progress": weights.w_return_home_progress * return_home_progress,
        "collision": -weights.w_collision * collision.float(),
        "near_obstacle": -weights.w_near_obstacle * near_obstacle,
        "time": -weights.w_time * torch.ones_like(map_progress) * float(dt),
        "idle": -weights.w_idle * idle_mask.float(),
        "oscillation": -weights.w_oscillation * yaw_flip.float(),
        "action_smoothness": -weights.w_action_smoothness * smoothness,
        "altitude_error": -weights.w_altitude_error * altitude_error,
    }
    total = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    return total, terms


def combine_rewards(extrinsic, new_cell_reward, rnd_reward, semantic_novelty, weights: RewardWeights | None = None):
    weights = weights or RewardWeights()
    terms = {
        "extrinsic": extrinsic,
        "new_cell_count": weights.beta_count * new_cell_reward,
        "rnd": weights.beta_rnd * rnd_reward,
        "semantic_novelty": weights.beta_semantic_novelty * semantic_novelty,
    }
    total = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    return total, terms
