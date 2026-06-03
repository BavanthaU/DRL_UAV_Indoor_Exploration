from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class DepthLineSafetyShieldConfig:
    enabled: bool = True
    min_depth_fraction: float = 0.08
    front_fraction: float = 0.34


class DepthLineSafetyShield:
    """Suppress unsafe forward velocity from normalized depth-line observations."""

    def __init__(self, cfg: DepthLineSafetyShieldConfig | None = None):
        if torch is None:
            raise RuntimeError("DepthLineSafetyShield requires PyTorch.")
        self.cfg = cfg or DepthLineSafetyShieldConfig()

    def __call__(self, actions, obs: dict):
        if not self.cfg.enabled or "depth_line" not in obs:
            return actions, torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
        depth = obs["depth_line"].float()
        if depth.ndim != 2 or depth.shape[1] == 0:
            return actions, torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
        rays = depth.shape[1]
        span = max(1, int(rays * self.cfg.front_fraction))
        start = max(0, (rays - span) // 2)
        front_depth = depth[:, start : start + span].min(dim=1).values
        blocked_forward = (front_depth < self.cfg.min_depth_fraction) & (actions[:, 0] > 0.0)
        shielded = actions.clone()
        shielded[blocked_forward, 0] = 0.0
        interventions = (shielded != actions).any(dim=-1)
        return shielded, interventions
