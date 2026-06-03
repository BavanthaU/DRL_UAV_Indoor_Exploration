from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def crop_grid(grid, centers, crop_size: int):
    if torch is None:
        raise RuntimeError("crop_grid requires PyTorch.")
    batch, height, width = grid.shape
    half = crop_size // 2
    padded = torch.nn.functional.pad(grid, (half, half, half, half), value=0)
    crops = []
    for env_id in range(batch):
        row = int(centers[env_id, 0].item()) + half
        col = int(centers[env_id, 1].item()) + half
        crops.append(padded[env_id, row - half : row - half + crop_size, col - half : col - half + crop_size])
    return torch.stack(crops)


def frontier_mask_from_visited(visited):
    free = visited.bool()
    unknown = ~free
    frontier = torch.zeros_like(free)
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = torch.roll(unknown, shifts=(d_row, d_col), dims=(1, 2))
        frontier |= free & shifted
    frontier[:, 0, :] = False
    frontier[:, -1, :] = False
    frontier[:, :, 0] = False
    frontier[:, :, -1] = False
    return frontier


def depth_line_from_camera(depth_image, width: int = 64):
    if depth_image is None:
        return None
    if depth_image.ndim == 4:
        depth_image = depth_image.squeeze(-1)
    row = depth_image[:, depth_image.shape[1] // 2, :]
    row = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row = torch.nn.functional.interpolate(row.unsqueeze(1), size=width, mode="linear", align_corners=False).squeeze(1)
    return row.clamp(0.0, 6.0) / 6.0

