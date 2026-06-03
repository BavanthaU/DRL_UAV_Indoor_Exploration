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


def sustained_altitude_out_of_bounds(
    altitude,
    min_altitude: float,
    max_altitude: float,
    low_counter,
    high_counter,
    required_steps: int,
):
    """Track low/high altitude violations and terminate only after persistence."""

    if torch is None:
        raise RuntimeError("sustained_altitude_out_of_bounds requires PyTorch.")
    required_steps = max(1, int(required_steps))
    finite = torch.isfinite(altitude)
    low = finite & (altitude < float(min_altitude))
    high = finite & (altitude > float(max_altitude))
    low_counter = torch.where(low, low_counter + 1, torch.zeros_like(low_counter))
    high_counter = torch.where(high, high_counter + 1, torch.zeros_like(high_counter))
    terminal = (~finite) | (low_counter >= required_steps) | (high_counter >= required_steps)
    return terminal, low, high, low_counter, high_counter
