from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class LocalDoneConfig:
    no_new_cells_for_s: float = 4.0
    local_new_cell_rate_threshold: float = 0.05
    min_open_depth_m: float = 0.60
    coverage_radius_saturation: float = 0.95
    repeated_cell_ratio_threshold: float = 0.75
    max_local_steps: int = 160


@dataclass
class LocalDoneResult:
    done: "torch.Tensor"
    reason: list[str]


class LocalDoneEvaluator:
    def __init__(self, cfg: LocalDoneConfig | None = None):
        self.cfg = cfg or LocalDoneConfig()

    def __call__(self, metrics: dict) -> LocalDoneResult:
        if torch is None:
            raise RuntimeError("LocalDoneEvaluator requires PyTorch.")
        reference = next(value for value in metrics.values() if hasattr(value, "shape"))
        batch = int(reference.shape[0])
        done = torch.zeros(batch, dtype=torch.bool, device=reference.device)
        reason = ["continue" for _ in range(batch)]

        checks = [
            ("no_new_cells_for_s", metrics.get("time_since_last_new_cell", 0.0) >= self.cfg.no_new_cells_for_s),
            ("low_new_cell_rate", metrics.get("local_new_cell_rate", 1.0) < self.cfg.local_new_cell_rate_threshold),
            ("open_depth_exhausted", metrics.get("max_open_depth", 10.0) < self.cfg.min_open_depth_m),
            (
                "coverage_radius_saturated",
                metrics.get("coverage_radius_fraction", 0.0) >= self.cfg.coverage_radius_saturation,
            ),
            (
                "repeated_cell_ratio",
                metrics.get("repeated_cell_ratio", 0.0) >= self.cfg.repeated_cell_ratio_threshold,
            ),
            ("max_local_steps", metrics.get("local_steps", 0) >= self.cfg.max_local_steps),
            ("stuck_detector", metrics.get("stuck", torch.zeros(batch, device=reference.device)).bool()),
            ("oscillation_detector", metrics.get("oscillation", torch.zeros(batch, device=reference.device)).bool()),
        ]
        for name, mask in checks:
            if not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, device=reference.device).expand(batch)
            mask = mask.bool()
            newly_done = mask & ~done
            if newly_done.any():
                for idx in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
                    reason[int(idx)] = name
            done |= mask
        return LocalDoneResult(done=done, reason=reason)
