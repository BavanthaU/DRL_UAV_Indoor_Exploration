from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def stack_observations(obs_buf: list[dict]):
    if torch is None:
        raise RuntimeError("stack_observations requires PyTorch.")
    keys = obs_buf[0].keys()
    return {key: torch.stack([obs[key] for obs in obs_buf]) for key in keys}


def flatten_hierarchical_rollout(rollout: dict):
    if torch is None:
        raise RuntimeError("flatten_hierarchical_rollout requires PyTorch.")
    return {
        "obs": {key: value.flatten(0, 1) for key, value in rollout["obs"].items()},
        "actions": rollout["actions"].flatten(0, 1),
        "options": rollout["options"].flatten(0, 1),
        "candidates": rollout["candidates"].flatten(0, 1),
        "log_probs": rollout["log_probs"].flatten(0, 1),
        "advantages": rollout["advantages"].flatten(0, 1),
        "returns": rollout["returns"].flatten(0, 1),
    }
