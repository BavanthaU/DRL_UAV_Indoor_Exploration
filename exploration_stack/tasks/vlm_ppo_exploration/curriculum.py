from __future__ import annotations


def linear_curriculum(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress

