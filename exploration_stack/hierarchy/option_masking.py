from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .option_types import ExplorationOption, NUM_OPTIONS


def build_option_mask(candidate_features, candidate_mask):
    """Build a numerically safe option mask for the categorical option head."""

    if torch is None:
        raise RuntimeError("build_option_mask requires PyTorch.")
    if candidate_mask.ndim != 2:
        raise ValueError("candidate_mask must be [B,N]")
    batch = candidate_mask.shape[0]
    mask = torch.ones(batch, NUM_OPTIONS, dtype=torch.bool, device=candidate_mask.device)
    no_candidate = ~candidate_mask.any(dim=1)
    if no_candidate.any():
        mask[no_candidate] = False
        mask[no_candidate, int(ExplorationOption.ROTATE_SCAN)] = True
    return mask
