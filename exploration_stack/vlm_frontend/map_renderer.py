from __future__ import annotations

from dataclasses import dataclass

from .base import torch


@dataclass
class MapRenderConfig:
    image_size: int = 224
    unknown_color: tuple[float, float, float] = (0.18, 0.18, 0.18)
    free_color: tuple[float, float, float] = (0.82, 0.82, 0.82)
    occupied_color: tuple[float, float, float] = (0.05, 0.05, 0.05)
    frontier_color: tuple[float, float, float] = (0.1, 0.55, 1.0)
    trajectory_color: tuple[float, float, float] = (1.0, 0.8, 0.15)
    robot_color: tuple[float, float, float] = (0.0, 1.0, 0.35)


class MapRenderer:
    """Render occupancy/frontier/trajectory crops to VLM-compatible RGB tensors."""

    def __init__(self, cfg: MapRenderConfig | None = None):
        if torch is None:
            raise RuntimeError("MapRenderer requires PyTorch.")
        self.cfg = cfg or MapRenderConfig()

    def render(
        self,
        occupancy: "torch.Tensor",
        *,
        frontier_mask: "torch.Tensor | None" = None,
        trajectory_mask: "torch.Tensor | None" = None,
        robot_xy: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        if occupancy.ndim == 2:
            occupancy = occupancy.unsqueeze(0)
        if occupancy.ndim != 3:
            raise ValueError(f"occupancy must be [B,H,W] or [H,W], got {tuple(occupancy.shape)}")
        batch, height, width = occupancy.shape
        image = torch.zeros(batch, 3, height, width, device=occupancy.device, dtype=torch.float32)
        for value, color in ((0, self.cfg.unknown_color), (1, self.cfg.free_color), (2, self.cfg.occupied_color)):
            mask = occupancy == value
            for channel, channel_value in enumerate(color):
                image[:, channel][mask] = channel_value
        if frontier_mask is not None:
            self._paint_mask(image, frontier_mask, self.cfg.frontier_color)
        if trajectory_mask is not None:
            self._paint_mask(image, trajectory_mask, self.cfg.trajectory_color)
        if robot_xy is not None:
            self._paint_robot(image, robot_xy)
        if height != self.cfg.image_size or width != self.cfg.image_size:
            image = torch.nn.functional.interpolate(
                image,
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="nearest",
            )
        return image

    def _paint_mask(self, image, mask, color: tuple[float, float, float]) -> None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.bool()
        for channel, channel_value in enumerate(color):
            image[:, channel][mask] = channel_value

    def _paint_robot(self, image, robot_xy) -> None:
        batch, _, height, width = image.shape
        for env_id in range(batch):
            row = int(robot_xy[env_id, 1].item())
            col = int(robot_xy[env_id, 0].item())
            row0, row1 = max(0, row - 1), min(height, row + 2)
            col0, col1 = max(0, col - 1), min(width, col + 2)
            for channel, channel_value in enumerate(self.cfg.robot_color):
                image[env_id, channel, row0:row1, col0:col1] = channel_value

