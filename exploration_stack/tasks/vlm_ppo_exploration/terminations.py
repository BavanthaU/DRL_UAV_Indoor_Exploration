from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def local_map_complete(frontier_count, closed_counter, required_closed_steps: int):
    return (frontier_count <= 0) & (closed_counter >= required_closed_steps)


def altitude_out_of_bounds(altitude, min_altitude: float, max_altitude: float):
    nonfinite = ~torch.isfinite(altitude) if torch is not None else False
    return nonfinite | (altitude < min_altitude) | (altitude > max_altitude)
