from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from exploration_stack.vlm_frontend.map_renderer import MapRenderConfig, MapRenderer


@dataclass
class ExplorationMapSnapshot:
    """Agent-side explored map snapshot for evaluation visualization."""

    occupancy: "torch.Tensor"
    frontier_mask: "torch.Tensor | None"
    trajectory_mask: "torch.Tensor | None"
    robot_xy: "torch.Tensor | None"
    mapped_free_cells: float
    frontier_count: float
    env_id: int
    backend: str

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "env_id": self.env_id,
            "frontier_count": self.frontier_count,
            "mapped_free_cells": self.mapped_free_cells,
            "source": "agent_internal_map",
        }


@dataclass
class BestExplorationMap:
    snapshot: ExplorationMapSnapshot
    metadata: dict[str, Any]


class BestExplorationMapTracker:
    """Track the best explored-map snapshot during vectorized evaluation."""

    def __init__(self, env):
        self.env = env
        self.latest_snapshots = capture_vector_exploration_maps(env)
        self.best: BestExplorationMap | None = None

    def refresh(self) -> None:
        self.latest_snapshots = capture_vector_exploration_maps(self.env)

    def record_completed(
        self,
        *,
        env_id: int,
        episode_index: int,
        episode_return: float,
        mapped_free_cells: float,
        episode_steps: int,
        global_step: int,
    ) -> None:
        snapshot = self.latest_snapshots[env_id]
        if snapshot is None:
            return
        mapped_free_cells = max(float(mapped_free_cells), snapshot.mapped_free_cells)
        metadata = {
            **snapshot.metadata(),
            "completed": True,
            "episode_index": episode_index,
            "episode_return": episode_return,
            "episode_steps": episode_steps,
            "global_step": global_step,
            "mapped_free_cells": mapped_free_cells,
            "selection": "max_mapped_free_cells_then_return",
        }
        self._maybe_update(snapshot, metadata)

    def record_best_partial(self, episode_returns, episode_steps, *, global_step: int) -> None:
        for env_id, snapshot in enumerate(self.latest_snapshots):
            if snapshot is None:
                continue
            episode_return = _tensor_item(episode_returns[env_id])
            metadata = {
                **snapshot.metadata(),
                "completed": False,
                "episode_index": None,
                "episode_return": episode_return,
                "episode_steps": int(_tensor_item(episode_steps[env_id])),
                "global_step": global_step,
                "mapped_free_cells": snapshot.mapped_free_cells,
                "selection": "partial_max_mapped_free_cells_then_return",
            }
            self._maybe_update(snapshot, metadata)

    def _maybe_update(self, snapshot: ExplorationMapSnapshot, metadata: dict[str, Any]) -> None:
        if self.best is None or _is_better(metadata, self.best.metadata):
            self.best = BestExplorationMap(snapshot=snapshot, metadata=metadata)


def capture_vector_exploration_maps(env) -> list[ExplorationMapSnapshot | None]:
    if torch is None:
        raise RuntimeError("Exploration map export requires PyTorch.")
    return [capture_exploration_map(env, env_id) for env_id in range(int(env.num_envs))]


def capture_exploration_map(env, env_id: int) -> ExplorationMapSnapshot | None:
    """Capture the agent's current map memory without using hidden environment priors."""

    base_env = _unwrap_env(env)
    if hasattr(base_env, "_known_map"):
        return _capture_debug_map(base_env, env_id)
    if hasattr(base_env, "_visited"):
        return _capture_isaac_map(base_env, env_id)
    return None


def save_exploration_map_png(
    snapshot: ExplorationMapSnapshot,
    path: str | Path,
    *,
    image_size: int = 512,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save an RGB visualization of an explored-map snapshot."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Saving explored-map PNGs requires Pillow. Install `Pillow>=10.0`.") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = MapRenderer(MapRenderConfig(image_size=image_size))
    robot_xy = snapshot.robot_xy.unsqueeze(0) if snapshot.robot_xy is not None else None
    rendered = renderer.render(
        snapshot.occupancy.unsqueeze(0),
        frontier_mask=snapshot.frontier_mask.unsqueeze(0) if snapshot.frontier_mask is not None else None,
        trajectory_mask=snapshot.trajectory_mask.unsqueeze(0) if snapshot.trajectory_mask is not None else None,
        robot_xy=robot_xy,
    )[0]
    array = rendered.detach().cpu().clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(array)
    image = _with_legend(image, metadata or snapshot.metadata())
    image.save(output)
    return output


def write_exploration_map_metadata(
    snapshot: ExplorationMapSnapshot,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.metadata()
    if extra:
        payload.update(extra)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _capture_debug_map(env, env_id: int) -> ExplorationMapSnapshot:
    occupancy = env._known_map[env_id].detach().clone().to("cpu").long()
    frontier = env._frontier_mask()[env_id].detach().clone().to("cpu").bool()
    trajectory = env._trajectory[env_id].detach().clone().to("cpu").bool()
    center = env._centers()[env_id].detach().clone().to("cpu").float()
    robot_xy = torch.stack((center[1], center[0]))
    return ExplorationMapSnapshot(
        occupancy=occupancy,
        frontier_mask=frontier,
        trajectory_mask=trajectory,
        robot_xy=robot_xy,
        mapped_free_cells=float((occupancy == 1).sum().item()),
        frontier_count=float(frontier.sum().item()),
        env_id=int(env_id),
        backend="debug",
    )


def _capture_isaac_map(env, env_id: int) -> ExplorationMapSnapshot:
    from exploration_stack.tasks.vlm_ppo_exploration.observations import frontier_mask_from_visited

    visited = env._visited.detach().clone()
    occupancy = visited[env_id].to("cpu").long()
    frontier = frontier_mask_from_visited(visited)[env_id].detach().clone().to("cpu").bool()
    trajectory = env._trajectory[env_id].detach().clone().to("cpu").bool() if hasattr(env, "_trajectory") else None
    if hasattr(env, "_safe_root_pos_w"):
        centers = env._world_to_grid(env._safe_root_pos_w()[:, :2])
    else:
        centers = env._world_to_grid(env._robot.data.root_pos_w[:, :2])
    center = centers[env_id].detach().clone().to("cpu").float()
    robot_xy = torch.stack((center[1], center[0]))
    return ExplorationMapSnapshot(
        occupancy=occupancy,
        frontier_mask=frontier,
        trajectory_mask=trajectory,
        robot_xy=robot_xy,
        mapped_free_cells=float(occupancy.sum().item()),
        frontier_count=float(frontier.sum().item()),
        env_id=int(env_id),
        backend="isaac",
    )


def _unwrap_env(env):
    base_env = getattr(env, "env", env)
    return getattr(base_env, "unwrapped", base_env)


def _is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_key = (float(candidate.get("mapped_free_cells", 0.0)), float(candidate.get("episode_return", 0.0)))
    current_key = (float(current.get("mapped_free_cells", 0.0)), float(current.get("episode_return", 0.0)))
    return candidate_key > current_key


def _tensor_item(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def _with_legend(image, metadata: dict[str, Any]):
    from PIL import Image, ImageDraw

    legend_height = 96
    canvas = Image.new("RGB", (image.width, image.height + legend_height), color=(245, 245, 245))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    labels = [
        ("unknown", (46, 46, 46)),
        ("observed free", (209, 209, 209)),
        ("observed obstacle", (13, 13, 13)),
        ("frontier", (26, 140, 255)),
        ("trajectory", (255, 204, 38)),
        ("robot", (0, 255, 89)),
    ]
    x, y = 12, image.height + 12
    for label, color in labels:
        draw.rectangle((x, y, x + 14, y + 14), fill=color)
        draw.text((x + 20, y - 1), label, fill=(20, 20, 20))
        x += 118
        if x + 110 > image.width:
            x = 12
            y += 24
    summary = (
        f"mapped_free_cells={metadata.get('mapped_free_cells', 0):.0f}  "
        f"frontier_count={metadata.get('frontier_count', 0):.0f}  "
        f"return={metadata.get('episode_return', 0.0):.3f}"
    )
    draw.text((12, image.height + legend_height - 24), summary, fill=(20, 20, 20))
    return canvas
