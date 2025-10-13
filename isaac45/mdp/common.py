from __future__ import annotations

import torch
from typing import Optional


def get_raycast_planar_min_distance(
    env,
    *,
    vertical_tolerance: Optional[float] = 1.5,
    fallback_distance: Optional[float] = None,
) -> torch.Tensor:
    """Return the minimum planar distance to a detected obstacle from the ray-cast sensor.

    Args:
        env: Isaac Lab environment providing access to the scene graph.
        vertical_tolerance: Optional maximum allowed absolute vertical separation (in meters) between
            the sensor and the hit point for it to be considered valid. If ``None`` the vertical
            component is ignored and all finite hits are considered.
        fallback_distance: Distance to use when no valid hit is detected. If ``None`` the sensor
            ``max_distance`` parameter is used (or 1e6 as a final fallback).

    Returns:
        Tensor with shape ``(num_envs,)`` containing the minimum planar distance for each environment.
    """
    device = env.device
    try:
        sensor = env.scene["ray_caster"]
    except KeyError:
        fill_value = fallback_distance or 1e6
        return torch.full((env.num_envs,), fill_value, device=device, dtype=torch.float32)

    ray_hits = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = sensor.data.pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    delta = ray_hits - sensor_pos

    planar = torch.linalg.norm(delta[..., :2], dim=-1)
    vertical = torch.abs(delta[..., 2])

    finite_mask = torch.isfinite(planar)
    if vertical_tolerance is not None:
        finite_mask &= torch.isfinite(vertical) & (vertical <= vertical_tolerance)

    max_distance = fallback_distance or getattr(sensor.cfg, "max_distance", 1e6)
    planar = torch.where(finite_mask, planar, torch.full_like(planar, max_distance))
    min_planar = planar.min(dim=1).values
    if getattr(env, "raycast_debug_print", False):
        print(f"[RayCaster] min planar distances: {min_planar.detach().cpu().tolist()}")
        if getattr(env, "raycast_debug_print_hits", False):
            hits_cpu = sensor.data.ray_hits_w.detach().cpu()
            print(f"[RayCaster] ray hits (env 0): {hits_cpu[0].tolist() if hits_cpu.numel() else []}")
    return min_planar


def ensure_ray_caster_initialized(env) -> None:
    """Ensure the ray-cast sensor has run its initialization prior to use."""
    base_env = getattr(env, "unwrapped", env)
    sensor = getattr(base_env.scene, "get", None)
    if sensor is None:
        return
    ray = base_env.scene.get("ray_caster", None)
    if ray is None or ray.is_initialized:
        return
    ray._initialize_impl()
    ray._is_initialized = True  # mark initialized so base class knows
    ray.reset()
