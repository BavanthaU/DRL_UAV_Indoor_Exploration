from __future__ import annotations

from typing import Any


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None


if nn is not None:

    class SemanticPriorNet(nn.Module):
        """Small Jetson-oriented frontier prior network skeleton.

        Inputs are intentionally compact: local occupancy crop, semantic line,
        depth line, candidate frontier features, and room-graph features.
        """

        def __init__(
            self,
            *,
            occupancy_channels: int = 3,
            semantic_line_dim: int = 64,
            depth_line_dim: int = 64,
            frontier_feature_dim: int = 8,
            room_graph_feature_dim: int = 16,
            hidden_dim: int = 128,
            max_frontiers: int = 16,
        ):
            super().__init__()
            self.max_frontiers = max_frontiers
            self.map_encoder = nn.Sequential(
                nn.Conv2d(occupancy_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            side_dim = semantic_line_dim + depth_line_dim + frontier_feature_dim + room_graph_feature_dim
            self.head = nn.Sequential(
                nn.Linear(32 * 4 * 4 + side_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 5),
            )

        def forward(
            self,
            occupancy_crop: "torch.Tensor",
            semantic_line: "torch.Tensor",
            depth_line: "torch.Tensor",
            frontier_features: "torch.Tensor",
            room_graph_features: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            encoded_map = self.map_encoder(occupancy_crop)
            x = torch.cat([encoded_map, semantic_line, depth_line, frontier_features, room_graph_features], dim=-1)
            raw = self.head(x)
            return {
                "frontier_score": torch.sigmoid(raw[..., 0]),
                "doorway_likelihood": torch.sigmoid(raw[..., 1]),
                "corridor_likelihood": torch.sigmoid(raw[..., 2]),
                "risk": torch.sigmoid(raw[..., 3]),
                "uncertainty": torch.sigmoid(raw[..., 4]),
            }

else:

    class SemanticPriorNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError(
                "SemanticPriorNet requires PyTorch. Install torch only for "
                "distillation/training/export; the four-layer smoke tests do "
                "not require it."
            )
