from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def approximate_path_features(candidate_xy, map_size: int):
    if torch is None:
        raise RuntimeError("approximate_path_features requires PyTorch.")
    distance = torch.linalg.norm(candidate_xy, dim=-1)
    path_cost = distance / max(1.0, float(map_size))
    reachable = torch.isfinite(path_cost).float()
    clearance = torch.clamp(1.0 - path_cost, 0.0, 1.0)
    return torch.stack([path_cost, reachable, clearance], dim=-1)
