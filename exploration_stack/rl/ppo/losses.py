from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


def ppo_clipped_policy_loss(new_log_prob, old_log_prob, advantages, clip_param: float):
    if torch is None:
        raise RuntimeError("ppo_clipped_policy_loss requires PyTorch.")
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    return -torch.min(unclipped, clipped).mean()


def value_mse_loss(value, returns):
    if F is None:
        raise RuntimeError("value_mse_loss requires PyTorch.")
    return F.mse_loss(value, returns)
