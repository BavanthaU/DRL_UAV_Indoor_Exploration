from __future__ import annotations

from pathlib import Path

from exploration_stack.core import ExplorationMap, Pose3D, SensorPacket, SlamState
from exploration_stack.slam.base import SlamBackend


class IsaacRosVisualSlamBackend(SlamBackend):
    """ROS 2 wrapper shell for Isaac ROS Visual SLAM / cuVSLAM."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        try:
            import rclpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "IsaacRosVisualSlamBackend requires ROS 2 Python bindings and "
                "Isaac ROS Visual SLAM. Install/activate ROS 2 and Isaac ROS, "
                "or set slam_backend=sim_ground_truth_debug/existing_fallback "
                "for smoke tests."
            ) from exc
        raise NotImplementedError(
            "IsaacRosVisualSlamBackend is a topic-wrapper skeleton. Configure "
            "camera/imu topics, TF frames, and cuVSLAM output before use."
        )

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, sensor_packet: SensorPacket) -> SlamState:
        raise NotImplementedError

    def get_pose(self) -> Pose3D:
        raise NotImplementedError

    def get_map(self) -> ExplorationMap | None:
        raise NotImplementedError

    def save_map(self, path: str | Path) -> None:
        raise NotImplementedError

    def load_map(self, path: str | Path) -> None:
        raise NotImplementedError
