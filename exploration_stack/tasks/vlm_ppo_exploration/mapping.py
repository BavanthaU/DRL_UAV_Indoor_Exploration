from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def depth_line_samples(depth_image, *, width: int, max_depth_m: float, min_depth_m: float = 0.1):
    """Sample the center depth row as metric ray ranges and endpoint-hit flags."""

    if torch is None:
        raise RuntimeError("depth_line_samples requires PyTorch.")
    if depth_image is None:
        return None, None
    if depth_image.ndim == 4:
        depth_image = depth_image.squeeze(-1)
    if depth_image.ndim != 3:
        raise ValueError("depth_image must be [B,H,W] or [B,H,W,1]")
    row = depth_image[:, depth_image.shape[1] // 2, :]
    columns = torch.linspace(0, row.shape[1] - 1, width, device=row.device).round().long()
    samples = row[:, columns].float()
    hit_mask = torch.isfinite(samples) & (samples >= float(min_depth_m)) & (samples < float(max_depth_m) - 1.0e-3)
    distances = torch.nan_to_num(samples, nan=float(max_depth_m), posinf=float(max_depth_m), neginf=0.0)
    distances = distances.clamp(float(min_depth_m), float(max_depth_m))
    return distances, hit_mask


def integrate_depth_line_occupancy(
    occupancy,
    trajectory,
    centers,
    yaw,
    distances_m,
    hit_mask,
    *,
    resolution_m: float,
    max_range_m: float,
    horizontal_fov_rad: float,
    free_value: int = 1,
    occupied_value: int = 2,
):
    """Integrate 2D depth rays into an unknown/free/occupied occupancy grid."""

    if torch is None:
        raise RuntimeError("integrate_depth_line_occupancy requires PyTorch.")
    if occupancy.ndim != 3:
        raise ValueError("occupancy must be [B,H,W]")
    if distances_m.ndim != 2 or hit_mask.shape != distances_m.shape:
        raise ValueError("distances_m and hit_mask must be [B,R]")
    batch, height, width = occupancy.shape
    if centers.shape[0] != batch or yaw.shape[0] != batch or distances_m.shape[0] != batch:
        raise ValueError("batch dimensions must match")

    ray_count = distances_m.shape[1]
    device = occupancy.device
    ray_angles = torch.linspace(
        -0.5 * float(horizontal_fov_rad),
        0.5 * float(horizontal_fov_rad),
        ray_count,
        device=device,
    )
    step = max(float(resolution_m) * 0.75, 1.0e-3)
    step_distances = torch.arange(0.0, float(max_range_m) + step, step, device=device)
    cell_scale = 1.0 / float(resolution_m)

    for env_id in range(batch):
        center = centers[env_id].to(device=device).float()
        occupancy[env_id, int(center[0].item()), int(center[1].item())] = free_value
        if trajectory is not None:
            trajectory[env_id, int(center[0].item()), int(center[1].item())] = True

        angles = yaw[env_id] + ray_angles
        ray_rows = torch.cos(angles)
        ray_cols = torch.sin(angles)
        free_distance = distances_m[env_id].clamp_min(0.0)
        free_distance = torch.where(hit_mask[env_id], (free_distance - step).clamp_min(0.0), free_distance)
        rows = center[0] + ray_rows[:, None] * step_distances[None, :] * cell_scale
        cols = center[1] + ray_cols[:, None] * step_distances[None, :] * cell_scale
        rows = rows.round().long()
        cols = cols.round().long()
        free_mask = step_distances[None, :] <= free_distance[:, None]
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        valid_free = free_mask & in_bounds
        occupancy[env_id, rows[valid_free], cols[valid_free]] = free_value

        valid_hits = hit_mask[env_id]
        if valid_hits.any():
            hit_rows = (center[0] + ray_rows[valid_hits] * distances_m[env_id, valid_hits] * cell_scale).round().long()
            hit_cols = (center[1] + ray_cols[valid_hits] * distances_m[env_id, valid_hits] * cell_scale).round().long()
            hit_in_bounds = (hit_rows >= 0) & (hit_rows < height) & (hit_cols >= 0) & (hit_cols < width)
            occupancy[env_id, hit_rows[hit_in_bounds], hit_cols[hit_in_bounds]] = occupied_value

    return occupancy
