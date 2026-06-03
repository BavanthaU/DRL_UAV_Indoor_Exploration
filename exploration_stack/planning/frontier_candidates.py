from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def frontier_centroids_from_mask(frontier_mask, *, max_candidates: int):
    if torch is None:
        raise RuntimeError("frontier_centroids_from_mask requires PyTorch.")
    if frontier_mask.ndim != 3:
        raise ValueError("frontier_mask must be [B,H,W]")
    batch, height, width = frontier_mask.shape
    centroids = torch.zeros(batch, max_candidates, 2, device=frontier_mask.device)
    valid = torch.zeros(batch, max_candidates, dtype=torch.bool, device=frontier_mask.device)
    center = torch.tensor([height // 2, width // 2], device=frontier_mask.device)
    for env_id in range(batch):
        cells = torch.nonzero(frontier_mask[env_id] > 0, as_tuple=False)
        if cells.numel() == 0:
            continue
        cells = cells.float()
        distances = torch.linalg.norm(cells - center.float(), dim=1)
        order = torch.argsort(distances)
        selected = cells[order[:max_candidates]]
        count = selected.shape[0]
        centroids[env_id, :count] = selected
        valid[env_id, :count] = True
    return centroids, valid
