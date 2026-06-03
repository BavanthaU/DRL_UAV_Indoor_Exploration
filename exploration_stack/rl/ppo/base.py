from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PPOTrainResult:
    timesteps: int
    updates: int
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None


class TrainerAdapter(ABC):
    """Backend-neutral PPO training interface."""

    @abstractmethod
    def train(self, env, *, max_iterations: int) -> PPOTrainResult:
        """Train PPO for a bounded number of iterations."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save trainer/model state."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trainer/model state."""

