from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class RewardWeights:
    success_threshold: float = 0.85
    w_success: float = 10.0
    w_frontier_progress: float = 0.2
    w_global_coverage_progress: float = 2.0
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
    coverage_ratio,
    prev_coverage_ratio,
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
    weights: RewardWeights | None = None,
):
    weights = weights or RewardWeights()
    if frontier_distance_delta is None:
        frontier_distance_delta = torch.zeros_like(coverage_ratio)
    success = (coverage_ratio >= weights.success_threshold).float()
    delta_coverage = (coverage_ratio - prev_coverage_ratio).clamp_min(0.0)
    near_obstacle = (float(safety_radius) - esdf_clearance).clamp_min(0.0)
    smoothness = torch.sum(torch.square(action - prev_action), dim=-1)
    altitude_error = torch.abs(altitude - target_altitude)
    terms = {
        "success": weights.w_success * success,
        "frontier_progress": weights.w_frontier_progress * frontier_distance_delta,
        "global_coverage_progress": weights.w_global_coverage_progress * delta_coverage,
        "collision": -weights.w_collision * collision.float(),
        "near_obstacle": -weights.w_near_obstacle * near_obstacle,
        "time": -weights.w_time * torch.ones_like(coverage_ratio) * float(dt),
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
