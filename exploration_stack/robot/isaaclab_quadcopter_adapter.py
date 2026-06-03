from __future__ import annotations

from typing import Any

from exploration_stack.core import DroneState, LocalCommand, SensorPacket
from exploration_stack.robot.base import RobotAdapter


class IsaacLabQuadcopterAdapter(RobotAdapter):
    """Skeleton for Isaac Lab's quadcopter task/template path.

    The current repository still uses a custom ideal 2D action term for the
    active training setup. This adapter is the future integration point for a
    proper Isaac Lab quadrotor/controller backend.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        try:
            import isaaclab  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "IsaacLabQuadcopterAdapter requires Isaac Lab. Activate the "
                "Isaac Sim 5.1 / IsaacLab 2.3.2 conda environment before using "
                "robot_backend=isaaclab_quadcopter."
            ) from exc
        raise NotImplementedError(
            "IsaacLabQuadcopterAdapter is a skeleton. Wire it to Isaac Lab's "
            "quadcopter task/template before selecting it for training."
        )

    def reset(self) -> DroneState:
        raise NotImplementedError

    def step(self, action: LocalCommand) -> DroneState:
        raise NotImplementedError

    def get_state(self) -> DroneState:
        raise NotImplementedError

    def get_sensors(self) -> SensorPacket:
        raise NotImplementedError

    def apply_local_command(self, cmd: LocalCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
